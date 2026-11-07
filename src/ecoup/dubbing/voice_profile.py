"""Voice profiling and clean reference audio extraction (S4)."""

from __future__ import annotations

import json
import os
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from .asr import DubbingSegment, load_wav_float32, resample_audio
from .diarization import _write_wav_pcm16


@dataclass
class VoiceProfile:
    speaker_id: str
    character_name: str
    quality_score: float  # 0.0 to 1.0
    is_good_reference: bool
    reference_wav: str
    total_segments: int
    speaking_duration_s: float
    flags: list[str] = field(default_factory=list)
    reference_text: str = ""


def calculate_audio_quality(audio: np.ndarray, sr: int = 16000) -> tuple[float, list[str]]:
    """Compute audio quality score and warning flags for a vocal sample."""
    flags = []
    if len(audio) == 0:
        return 0.1, ["sample_empty"]

    dur = len(audio) / sr

    if dur < 1.5:
        flags.append("sample_too_short")
    elif dur > 12.0:
        flags.append("sample_too_long")

    # RMS energy
    rms = float(np.sqrt(np.mean(audio**2)))
    if rms < 0.01:
        flags.append("low_energy_or_whisper")
    elif rms > 0.4:
        flags.append("potential_clipping")

    # Signal to noise ratio proxy (energy vs silent frames)
    frame_size = int(0.02 * sr)  # 20ms
    num_frames = len(audio) // frame_size
    if num_frames > 0:
        frame_energies = [
            float(np.mean(audio[i * frame_size : (i + 1) * frame_size] ** 2))
            for i in range(num_frames)
        ]
        sorted_e = sorted(frame_energies)
        noise_floor = sorted_e[int(num_frames * 0.1)]  # 10th percentile
        peak_speech = sorted_e[int(num_frames * 0.9)]  # 90th percentile
        snr = peak_speech / (noise_floor + 1e-8)
    else:
        snr = 1.0

    score = 1.0
    if "sample_too_short" in flags:
        score -= 0.4
    if "low_energy_or_whisper" in flags:
        score -= 0.2
    if snr < 5.0:
        flags.append("noisy_background")
        score -= 0.25

    score = max(0.1, min(1.0, round(score, 2)))
    return score, flags


def build_voice_profiles(
    dialogue_wav: str | Path,
    segments: Sequence[DubbingSegment],
    output_dir: str | Path,
    target_sr: int = 24000,
) -> dict[str, VoiceProfile]:
    """Extract clean reference audio and build VoiceProfile for each speaker."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dialogue_wav = Path(dialogue_wav)

    full_audio = load_wav_float32(dialogue_wav, target_sr=16000)
    sr_load = 16000

    speaker_segments: dict[str, list[DubbingSegment]] = {}
    for s in segments:
        spk = s.speaker_id or "spk_0"
        speaker_segments.setdefault(spk, []).append(s)

    profiles: dict[str, VoiceProfile] = {}

    for spk, segs in speaker_segments.items():
        spk_dir = output_dir / spk
        spk_dir.mkdir(parents=True, exist_ok=True)
        ref_file = spk_dir / "ref_clean.wav"

        # Prioritize clean non-overlapping segments between 2s and 6s
        candidates = [s for s in segs if not s.is_overlap and 2.0 <= s.duration <= 7.0]
        if not candidates:
            # Fallback to any non-overlap segment
            candidates = [s for s in segs if not s.is_overlap and s.duration >= 1.0]
        if not candidates:
            candidates = list(segs)

        # Select best segment or combine 2 short segments
        selected_audio_chunks: list[np.ndarray] = []
        selected_texts: list[str] = []
        cur_dur = 0.0
        for cand in candidates:
            st = int(cand.start * sr_load)
            et = int(cand.end * sr_load)
            chunk = full_audio[st:et]
            selected_audio_chunks.append(chunk)
            if cand.text and cand.text.strip():
                selected_texts.append(cand.text.strip())
            cur_dur += cand.duration
            if cur_dur >= 4.0:
                break

        ref_prompt_text = " ".join(selected_texts)

        if selected_audio_chunks:
            combined = np.concatenate(selected_audio_chunks)
        else:
            combined = np.zeros(int(2.0 * sr_load), dtype=np.float32)

        # Resample to target_sr (24kHz for modern TTS vocoders)
        ref_audio = resample_audio(combined, sr_load, target_sr)
        _write_wav_pcm16(ref_file, ref_audio, target_sr)

        quality_score, flags = calculate_audio_quality(combined, sr=sr_load)
        is_good = quality_score >= 0.6 and "sample_too_short" not in flags

        total_dur = round(sum(s.duration for s in segs), 2)
        vp = VoiceProfile(
            speaker_id=spk,
            character_name=f"Character {spk.replace('spk_', '#')}",
            quality_score=quality_score,
            is_good_reference=is_good,
            reference_wav=str(ref_file.relative_to(output_dir.parent)),
            total_segments=len(segs),
            speaking_duration_s=total_dur,
            flags=flags,
            reference_text=ref_prompt_text,
        )

        # Save profile.json
        profile_json = spk_dir / "profile.json"
        with open(profile_json, "w", encoding="utf-8") as f:
            json.dump(asdict(vp), f, ensure_ascii=False, indent=2)

        profiles[spk] = vp

    return profiles
