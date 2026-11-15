"""Persistent Voice Bank & multi-speaker cloning utilities for Tab 4 (Voice Cloning Studio)."""

from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from .asr import load_wav_float32
from .config import DubbingConfig
from .tts import (
    MockVoiceBackend,
    OpenVoiceBackend,
    VoiceBackend,
    estimate_median_f0_hz,
    synthesize_text_to_mp3,
)


SPECIAL_SELF = "__self__"
SPECIAL_ORIGINAL = "__original__"


def default_voice_bank_dir() -> Path:
    """Return the default Voice Bank storage directory for Colab or local runs."""
    if Path("/content").exists():
        return Path("/content/voice_bank")
    return Path("outputs/voice_bank")


def _slugify(text: str, fallback: str = "voice") -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", (text or "").strip()).strip("_").lower()
    return (cleaned[:32] if cleaned else fallback)


@dataclass
class ClonedVoiceEntry:
    clone_id: str
    display_name: str
    source_video: str
    speaker_id: str
    clone_index: int = 1
    quality_score: float = 0.0
    speaking_duration_s: float = 0.0
    estimated_f0_hz: float = 0.0
    reference_text: str = ""
    ref_wav_path: str = ""
    embedding_path: str = ""
    preview_mp3_path: str = ""
    created_at: str = ""
    batch_ts: float = 0.0


class VoiceBankManager:
    """Manage cloned speaker profiles (.wav + .pt + .mp3 + metadata) on disk."""

    META_FILENAME = "clone_meta.json"
    REF_FILENAME = "ref_clean.wav"
    EMB_FILENAME = "embedding.pt"
    PREVIEW_FILENAME = "preview_vi.mp3"

    def __init__(self, bank_dir: Optional[str | Path] = None):
        self.bank_dir = Path(bank_dir) if bank_dir else default_voice_bank_dir()
        self.bank_dir.mkdir(parents=True, exist_ok=True)

    def _entry_dir(self, clone_id: str) -> Path:
        safe_id = Path(clone_id).name
        return self.bank_dir / safe_id

    def _hydrate_entry(self, entry_dir: Path, data: dict[str, Any]) -> ClonedVoiceEntry:
        ref_file = entry_dir / self.REF_FILENAME
        emb_file = entry_dir / self.EMB_FILENAME
        prev_file = entry_dir / self.PREVIEW_FILENAME
        return ClonedVoiceEntry(
            clone_id=str(data.get("clone_id", entry_dir.name)),
            display_name=str(data.get("display_name", entry_dir.name)),
            source_video=str(data.get("source_video", "")),
            speaker_id=str(data.get("speaker_id", "spk_0")),
            clone_index=int(data.get("clone_index", 1) or 1),
            quality_score=float(data.get("quality_score", 0.0) or 0.0),
            speaking_duration_s=float(data.get("speaking_duration_s", 0.0) or 0.0),
            estimated_f0_hz=float(data.get("estimated_f0_hz", 0.0) or 0.0),
            reference_text=str(data.get("reference_text", "")),
            ref_wav_path=str(ref_file) if ref_file.exists() else str(data.get("ref_wav_path", "")),
            embedding_path=str(emb_file) if emb_file.exists() else "",
            preview_mp3_path=str(prev_file) if prev_file.exists() else "",
            created_at=str(data.get("created_at", "")),
            batch_ts=float(data.get("batch_ts", 0.0) or 0.0),
        )

    def save_voice(
        self,
        clone_id: str,
        ref_wav_src: str | Path,
        display_name: Optional[str] = None,
        source_video: str = "",
        speaker_id: str = "spk_0",
        clone_index: int = 1,
        quality_score: float = 0.0,
        speaking_duration_s: float = 0.0,
        reference_text: str = "",
        estimated_f0_hz: Optional[float] = None,
        backend: Optional[VoiceBackend] = None,
        preview_text: str = "Xin chào, đây là giọng nói tiếng Việt được nhân bản.",
        batch_ts: Optional[float] = None,
        skip_preview: bool = False,
    ) -> ClonedVoiceEntry:
        """Persist a cloned speaker entry into the Voice Bank."""
        safe_id = _slugify(clone_id, fallback=f"clone_{int(time.time())}")
        entry_dir = self._entry_dir(safe_id)
        entry_dir.mkdir(parents=True, exist_ok=True)

        ref_src = Path(ref_wav_src)
        if not ref_src.exists():
            raise FileNotFoundError(f"Reference audio not found: {ref_src}")

        dest_ref = entry_dir / self.REF_FILENAME
        if ref_src.resolve() != dest_ref.resolve():
            shutil.copy2(ref_src, dest_ref)

        # Compute median F0 if not provided
        f0_hz = float(estimated_f0_hz) if estimated_f0_hz is not None else 0.0
        if f0_hz <= 0.0:
            try:
                wav_audio = load_wav_float32(dest_ref, target_sr=16000)
                f0_hz = estimate_median_f0_hz(wav_audio, 16000)
            except Exception:
                f0_hz = 0.0

        now_ts = float(batch_ts) if batch_ts is not None else time.time()
        created_iso = datetime.now(timezone.utc).isoformat()

        dest_emb = entry_dir / self.EMB_FILENAME
        dest_preview = entry_dir / self.PREVIEW_FILENAME

        # Write clone_meta.json BEFORE preview synthesis so OpenVoiceBackend can read estimated_f0_hz
        meta_payload = {
            "clone_id": safe_id,
            "display_name": display_name or safe_id,
            "source_video": source_video,
            "speaker_id": speaker_id,
            "clone_index": int(clone_index),
            "quality_score": round(float(quality_score), 3),
            "speaking_duration_s": round(float(speaking_duration_s), 2),
            "estimated_f0_hz": round(float(f0_hz), 1),
            "reference_text": reference_text,
            "created_at": created_iso,
            "batch_ts": now_ts,
        }
        meta_path = entry_dir / self.META_FILENAME
        meta_path.write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        try:
            # Extract OpenVoice speaker embedding (.pt) if backend supports it
            if backend is not None and hasattr(backend, "extract_speaker_embedding"):
                backend.extract_speaker_embedding(dest_ref, save_pt_path=dest_emb)

            # Generate Vietnamese MP3 sample so users can immediately audition the cloned voice
            if not skip_preview and backend is not None and (preview_text or "").strip():
                synthesize_text_to_mp3(
                    text=preview_text,
                    output_mp3=dest_preview,
                    backend=backend,
                    ref_audio_path=dest_ref,
                    speed=1.0,
                    speaker_id=safe_id,
                    bitrate="192k",
                )
        except Exception:
            shutil.rmtree(entry_dir, ignore_errors=True)
            raise

        return self._hydrate_entry(entry_dir, meta_payload)

    def get_voice(self, clone_id: str) -> Optional[ClonedVoiceEntry]:
        """Load one cloned voice by ID if it exists."""
        clean_id = self._normalize_choice_id(clone_id)
        if not clean_id:
            return None
        entry_dir = self._entry_dir(clean_id)
        meta_path = entry_dir / self.META_FILENAME
        ref_path = entry_dir / self.REF_FILENAME
        if not meta_path.exists() or not ref_path.exists():
            return None
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return self._hydrate_entry(entry_dir, data)

    def list_voices(self) -> list[ClonedVoiceEntry]:
        """Return all valid cloned voices, newest batch first and clone_index ascending."""
        if not self.bank_dir.exists():
            return []
        entries: list[ClonedVoiceEntry] = []
        for child in self.bank_dir.iterdir():
            if not child.is_dir() or child.name.startswith("_"):
                continue
            meta_path = child / self.META_FILENAME
            ref_path = child / self.REF_FILENAME
            if not meta_path.exists() or not ref_path.exists():
                continue
            try:
                data = json.loads(meta_path.read_text(encoding="utf-8"))
                entries.append(self._hydrate_entry(child, data))
            except Exception:
                continue

        entries.sort(key=lambda e: (-e.batch_ts, e.clone_index, e.clone_id))
        return entries

    def rename_voice(self, clone_id: str, new_name: str) -> ClonedVoiceEntry:
        """Update display_name in clone_meta.json."""
        entry = self.get_voice(clone_id)
        if entry is None:
            raise KeyError(f"Cloned voice not found: {clone_id}")
        clean_name = (new_name or "").strip()
        if not clean_name:
            raise ValueError("Tên hiển thị không được để trống.")
        meta_path = self._entry_dir(entry.clone_id) / self.META_FILENAME
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        data["display_name"] = clean_name
        meta_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return self._hydrate_entry(self._entry_dir(entry.clone_id), data)

    def delete_voice(self, clone_id: str) -> bool:
        """Remove a cloned voice directory from disk."""
        clean_id = self._normalize_choice_id(clone_id)
        if not clean_id:
            return False
        entry_dir = self._entry_dir(clean_id)
        if not entry_dir.exists():
            return False
        shutil.rmtree(entry_dir, ignore_errors=True)
        return not entry_dir.exists()

    @staticmethod
    def _normalize_choice_id(spec: str) -> str:
        """Accept raw clone_id or UI display strings like 'Name [clone_id]' or 'clone_id :: Name'."""
        text = (spec or "").strip()
        if not text:
            return ""
        if text in (SPECIAL_SELF, SPECIAL_ORIGINAL):
            return text
        if " :: " in text:
            return text.split(" :: ", 1)[0].strip()
        m = re.search(r"\[([a-zA-Z0-9_-]+)\]\s*$", text)
        if m:
            return m.group(1).strip()
        return text

    def resolve_voice(self, spec: str) -> Optional[Path]:
        """Resolve a clone specification (clone_id, UI label, dir, or wav path) to ref_clean.wav."""
        cleaned = self._normalize_choice_id(spec)
        if not cleaned or cleaned in (SPECIAL_SELF, SPECIAL_ORIGINAL):
            return None

        entry = self.get_voice(cleaned)
        if entry and entry.ref_wav_path and Path(entry.ref_wav_path).exists():
            return Path(entry.ref_wav_path)

        # Direct file or directory path
        cand = Path(spec.strip())
        if cand.is_file():
            return cand
        if cand.is_dir():
            ref_in_dir = cand / self.REF_FILENAME
            if ref_in_dir.is_file():
                return ref_in_dir

        # Exact match on display_name
        for v in self.list_voices():
            if v.display_name == spec.strip() and Path(v.ref_wav_path).exists():
                return Path(v.ref_wav_path)

        return None

    def format_choice_label(self, entry: ClonedVoiceEntry) -> str:
        """Format a consistent dropdown label containing clone_id for reliable lookup."""
        f0_tag = f" ~{entry.estimated_f0_hz:.0f}Hz" if entry.estimated_f0_hz > 0 else ""
        return f"{entry.display_name} ({entry.speaking_duration_s:.1f}s{f0_tag}) [{entry.clone_id}]"

    def get_dropdown_choices(self) -> list[str]:
        """Return formatted labels for UI dropdowns."""
        return [self.format_choice_label(v) for v in self.list_voices()]


def clone_voices_from_video(
    video_path: str | Path,
    bank_dir: Optional[str | Path] = None,
    project_dir: Optional[str | Path] = None,
    label_prefix: str = "",
    num_speakers: Optional[int] = None,
    min_speakers: int = 1,
    max_speakers: int = 8,
    distance_threshold: float = 0.72,
    whisper_model: str = "medium",
    source_lang: Optional[str] = None,
    min_speaker_duration_s: float = 0.8,
    device: str = "auto",
    pitch_match_mode: str = "auto",
    mock: bool = False,
    preview_text: str = "Xin chào, đây là giọng nói tiếng Việt được nhân bản từ nhân vật này.",
    on_progress: Optional[Callable[[str, float], None]] = None,
) -> list[ClonedVoiceEntry]:
    """Analyze a source video, extract each speaker's voice, and save clones to the Voice Bank."""
    from .pipeline import DubbingPipeline

    video_p = Path(video_path)
    if not video_p.exists():
        raise FileNotFoundError(f"Source video not found: {video_p}")

    bank = VoiceBankManager(bank_dir)
    if project_dir is None:
        safe_stem = _slugify(video_p.stem, fallback="src")
        project_dir = bank.bank_dir / "_projects" / safe_stem

    cfg = DubbingConfig()
    cfg.project_dir = str(project_dir)
    cfg.demucs.device = device
    cfg.demucs.mock = mock
    cfg.asr.model_size = whisper_model
    cfg.asr.source_lang = source_lang
    cfg.asr.mock = mock
    cfg.diarization.num_speakers = num_speakers
    cfg.diarization.min_speakers = min_speakers
    cfg.diarization.max_speakers = max_speakers
    cfg.diarization.distance_threshold = distance_threshold
    cfg.diarization.mock = mock
    cfg.tts.mock = mock
    cfg.tts.backend = "mock" if mock else "openvoice"
    cfg.tts.pitch_match_mode = pitch_match_mode

    pipeline = DubbingPipeline(video_p, config=cfg, on_progress=on_progress)
    diarized_segs = pipeline.run_s3_diarization()
    profiles = pipeline.run_s4_voice_profiles()

    if not profiles:
        raise RuntimeError("Không tìm thấy giọng nói nào trong video/audio nguồn.")

    # Order speakers consistently by first appearance time in the video
    first_seen: dict[str, float] = {}
    for seg in diarized_segs:
        spk = seg.speaker_id or "spk_0"
        if spk not in first_seen or seg.start < first_seen[spk]:
            first_seen[spk] = float(seg.start)

    ordered_spks = sorted(
        profiles.keys(),
        key=lambda s: (first_seen.get(s, 1e9), s),
    )

    # Filter out tiny noise clusters when multiple speakers exist
    if len(ordered_spks) > 1 and min_speaker_duration_s > 0:
        kept = [s for s in ordered_spks if profiles[s].speaking_duration_s >= min_speaker_duration_s]
        if kept:
            ordered_spks = kept
        else:
            ordered_spks = [max(ordered_spks, key=lambda s: profiles[s].speaking_duration_s)]

    backend: VoiceBackend = (
        MockVoiceBackend()
        if mock
        else OpenVoiceBackend(device=device, pitch_match_mode=pitch_match_mode)
    )

    batch_ts = time.time()
    ts_tag = datetime.now().strftime("%H%M%S")
    base_label = (label_prefix or "").strip() or video_p.stem
    slug_base = _slugify(base_label, fallback="clone")

    created_entries: list[ClonedVoiceEntry] = []
    total = len(ordered_spks)
    try:
        for idx, spk_id in enumerate(ordered_spks, start=1):
            if on_progress:
                on_progress("S4B_CLONE", (idx - 1) / max(1, total))

            prof = profiles[spk_id]
            ref_wav = pipeline.project_dir / prof.reference_wav
            if not ref_wav.exists():
                continue

            clone_id = f"{slug_base}_c{idx}_{ts_tag}"
            # Ensure uniqueness if called multiple times within 1 second
            suffix = 1
            while bank._entry_dir(clone_id).exists():
                clone_id = f"{slug_base}_c{idx}_{ts_tag}_{suffix}"
                suffix += 1

            display_name = f"Clone {idx} ({base_label})"
            entry = bank.save_voice(
                clone_id=clone_id,
                ref_wav_src=ref_wav,
                display_name=display_name,
                source_video=video_p.name,
                speaker_id=spk_id,
                clone_index=idx,
                quality_score=prof.quality_score,
                speaking_duration_s=prof.speaking_duration_s,
                reference_text=prof.reference_text,
                backend=backend,
                preview_text=preview_text,
                batch_ts=batch_ts,
            )
            created_entries.append(entry)
            if on_progress:
                on_progress("S4B_CLONE", idx / max(1, total))
    finally:
        # Release OpenVoice GPU converter if loaded
        if hasattr(backend, "_converter") and getattr(backend, "_converter") is not None:
            try:
                setattr(backend, "_converter", None)
                import gc
                import torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    return created_entries


def suggest_speaker_mapping(
    target_speakers: Sequence[dict[str, Any]],
    bank_voices: Sequence[ClonedVoiceEntry],
    mode: str = "order",
) -> dict[str, str]:
    """Suggest mapping from target video speakers (Giọng 1, 2...) to Voice Bank clones.

    Args:
        target_speakers: Ordered list of speaker dicts from analyze_target_speakers().
        bank_voices: List of available ClonedVoiceEntry items.
        mode: "order" (Giọng 1->Clone 1, Giọng 2->Clone 2) or "f0" (closest vocal pitch).
    """
    if not target_speakers:
        return {}
    if not bank_voices:
        return {str(s["speaker_id"]): SPECIAL_SELF for s in target_speakers}

    mapping: dict[str, str] = {}
    if mode == "f0":
        available = list(bank_voices)
        for spk in target_speakers:
            spk_id = str(spk["speaker_id"])
            tgt_f0 = float(spk.get("estimated_f0_hz", 0.0) or 0.0)
            pool = available if available else list(bank_voices)
            if tgt_f0 > 0:
                valid_f0 = [v for v in pool if v.estimated_f0_hz > 0]
                if valid_f0:
                    chosen = min(valid_f0, key=lambda v: abs(v.estimated_f0_hz - tgt_f0))
                else:
                    chosen = pool[0]
            else:
                chosen = pool[0]
            mapping[spk_id] = chosen.clone_id
            if chosen in available:
                available.remove(chosen)
        return mapping

    # Default "order" mode: Giọng 1 -> Clone 1, Giọng 2 -> Clone 2, ...
    for idx, spk in enumerate(target_speakers):
        spk_id = str(spk["speaker_id"])
        if idx < len(bank_voices):
            mapping[spk_id] = bank_voices[idx].clone_id
        elif len(bank_voices) == 1:
            mapping[spk_id] = bank_voices[0].clone_id
        else:
            mapping[spk_id] = SPECIAL_SELF
    return mapping
