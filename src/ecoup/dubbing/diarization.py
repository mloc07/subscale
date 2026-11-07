"""Speaker Diarization, Embedding extraction, and Clustering."""

from __future__ import annotations

import gc
import json
import os
import shutil
import subprocess
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from .asr import DubbingSegment, load_wav_float32


@dataclass
class DiarizationOptions:
    num_speakers: Optional[int] = None
    min_speakers: int = 1
    max_speakers: int = 10
    distance_threshold: float = 0.45  # Cosine distance threshold (1 - cos_sim)
    mock: bool = False


def _cluster_pure_numpy(emb_matrix: np.ndarray, distance_threshold: float, num_speakers: Optional[int] = None) -> list[int]:
    """Greedy clustering on cosine distance using pure numpy (no scipy/sklearn dependency needed)."""
    n = len(emb_matrix)
    if n <= 1:
        return [0] * n

    # Compute pairwise cosine distance: 1 - (u . v)
    dots = np.dot(emb_matrix, emb_matrix.T)
    dist = np.clip(1.0 - dots, 0.0, 2.0)

    clusters: list[list[int]] = []
    assigned = [-1] * n

    for i in range(n):
        if assigned[i] != -1:
            continue
        c_idx = len(clusters)
        clusters.append([i])
        assigned[i] = c_idx
        for j in range(i + 1, n):
            if assigned[j] == -1 and dist[i, j] <= distance_threshold:
                clusters[c_idx].append(j)
                assigned[j] = c_idx

    # If num_speakers is specified and we have too many clusters, merge closest
    if num_speakers is not None and len(clusters) > num_speakers:
        while len(clusters) > num_speakers:
            # Find two clusters with minimum distance
            min_d = 999.0
            best_pair = (0, 1)
            for c1 in range(len(clusters)):
                for c2 in range(c1 + 1, len(clusters)):
                    pts1 = clusters[c1]
                    pts2 = clusters[c2]
                    d_mean = dist[np.ix_(pts1, pts2)].mean()
                    if d_mean < min_d:
                        min_d = d_mean
                        best_pair = (c1, c2)
            c1, c2 = best_pair
            clusters[c1].extend(clusters[c2])
            clusters.pop(c2)

        # Reassign labels
        for c_idx, cluster_pts in enumerate(clusters):
            for pt in cluster_pts:
                assigned[pt] = c_idx

    return assigned


class SpeakerDiarizer:
    def __init__(self, options: Optional[DiarizationOptions] = None):
        self.options = options or DiarizationOptions()
        self._classifier = None

    def _get_classifier(self):
        if self._classifier is None and not self.options.mock:
            import torch
            from speechbrain.inference.speaker import EncoderClassifier

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._classifier = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device": device},
            )
        return self._classifier

    def extract_embedding(self, audio_chunk: np.ndarray, sr: int = 16000) -> np.ndarray:
        """Extract 192-dimensional ECAPA-TDNN embedding vector for an audio chunk."""
        if self.options.mock:
            # Deterministic mock embedding based on audio length/energy
            np.random.seed(int(abs(audio_chunk[:10].sum() * 1000) + len(audio_chunk)) % 10000)
            emb = np.random.randn(192).astype(np.float32)
            return emb / np.linalg.norm(emb)

        import torch

        classifier = self._get_classifier()
        device = classifier.device

        tensor = torch.from_numpy(audio_chunk).float().unsqueeze(0).to(device)
        with torch.no_grad():
            emb = classifier.encode_batch(tensor).squeeze().cpu().numpy()

        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb.astype(np.float32)

    def diarize_segments(
        self,
        dialogue_wav: str | Path,
        segments: Sequence[DubbingSegment],
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> list[DubbingSegment]:
        """Assign speaker_id to each segment via voice embedding clustering."""
        if not segments:
            return []

        dialogue_wav = Path(dialogue_wav)
        full_audio = load_wav_float32(dialogue_wav, target_sr=16000)
        sr = 16000

        # 1. Detect overlaps between segments
        for i in range(len(segments)):
            curr = segments[i]
            for j in range(i + 1, min(i + 4, len(segments))):
                nxt = segments[j]
                if nxt.start < curr.end - 0.05:  # Overlap tolerance 50ms
                    curr.is_overlap = True
                    nxt.is_overlap = True

        # 2. Extract embeddings for all valid segments
        embeddings: list[np.ndarray] = []
        valid_indices: list[int] = []

        for idx, seg in enumerate(segments):
            if should_cancel and should_cancel():
                break

            start_sample = int(seg.start * sr)
            end_sample = int(seg.end * sr)
            chunk = full_audio[start_sample:end_sample]

            # Require at least 0.3s of audio for meaningful embedding
            if len(chunk) >= int(0.3 * sr):
                emb = self.extract_embedding(chunk, sr=sr)
                embeddings.append(emb)
                valid_indices.append(idx)

        if not embeddings:
            for s in segments:
                s.speaker_id = "spk_0"
            return list(segments)

        # 3. Cluster embeddings
        emb_matrix = np.vstack(embeddings)
        num_samples = len(emb_matrix)

        if num_samples == 1:
            cluster_labels = [0]
        else:
            try:
                from scipy.cluster.hierarchy import fcluster, linkage
                from scipy.spatial.distance import pdist

                dist_condensed = pdist(emb_matrix, metric="cosine")
                dist_condensed = np.nan_to_num(dist_condensed, nan=1.0)
                dist_condensed = np.clip(dist_condensed, 0.0, 2.0)
                Z = linkage(dist_condensed, method="average")

                if self.options.num_speakers is not None:
                    k = max(1, min(self.options.num_speakers, num_samples))
                    cluster_labels = fcluster(Z, t=k, criterion="maxclust") - 1
                else:
                    cluster_labels = fcluster(Z, t=self.options.distance_threshold, criterion="distance") - 1
            except Exception:
                # Fallback to pure numpy clustering
                cluster_labels = _cluster_pure_numpy(
                    emb_matrix,
                    distance_threshold=self.options.distance_threshold,
                    num_speakers=self.options.num_speakers,
                )

        # Map labels back to segments
        for v_idx, label in zip(valid_indices, cluster_labels):
            segments[v_idx].speaker_id = f"spk_{int(label)}"

        # Fill any short skipped segments with nearest speaker
        last_spk = "spk_0"
        for s in segments:
            if s.speaker_id is not None:
                last_spk = s.speaker_id
            else:
                s.speaker_id = last_spk

        # Cleanup GPU classifier
        if self._classifier is not None:
            del self._classifier
            self._classifier = None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        return list(segments)


def _write_wav_pcm16(path: Path, audio: np.ndarray, sr: int = 16000) -> None:
    """Save float32 array [-1.0, 1.0] as 16-bit PCM WAV using standard library."""
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def export_speaker_previews(
    dialogue_wav: str | Path,
    segments: Sequence[DubbingSegment],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Export preview audio samples for each unique speaker."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dialogue_wav = Path(dialogue_wav)

    full_audio = load_wav_float32(dialogue_wav, target_sr=16000)
    sr = 16000

    speaker_segments: dict[str, list[DubbingSegment]] = {}
    for s in segments:
        spk = s.speaker_id or "spk_0"
        speaker_segments.setdefault(spk, []).append(s)

    previews: dict[str, Path] = {}
    for spk, segs in speaker_segments.items():
        spk_dir = output_dir / spk
        spk_dir.mkdir(parents=True, exist_ok=True)
        preview_file = spk_dir / "preview.wav"

        selected_audio: list[np.ndarray] = []
        total_len = 0
        for s in segs:
            if not s.is_overlap and s.duration >= 1.0:
                start_s = int(s.start * sr)
                end_s = int(s.end * sr)
                selected_audio.append(full_audio[start_s:end_s])
                total_len += (end_s - start_s)
                if total_len >= int(8.0 * sr):
                    break

        if not selected_audio:
            for s in segs[:3]:
                start_s = int(s.start * sr)
                end_s = int(s.end * sr)
                selected_audio.append(full_audio[start_s:end_s])

        if selected_audio:
            combined = np.concatenate(selected_audio)
            _write_wav_pcm16(preview_file, combined, sr)
            previews[spk] = preview_file

            profile_path = spk_dir / "profile.json"
            profile = {
                "speaker_id": spk,
                "segments_count": len(segs),
                "total_speaking_time": round(sum(s.duration for s in segs), 2),
                "preview_duration": round(len(combined) / sr, 2),
            }
            with open(profile_path, "w", encoding="utf-8") as f:
                json.dump(profile, f, indent=2)

    return previews
