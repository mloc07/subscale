"""Run report for the subtitle pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class SubtitleReport:
    video: str = ""
    detected_language: Optional[str] = None
    language_probability: float = 0.0
    skipped_translation: bool = False
    asr_segments: int = 0
    cues: int = 0
    translated_failed: int = 0
    translate_stats: dict = field(default_factory=dict)
    timings_s: dict = field(default_factory=dict)
    asr_peak_vram_gb: Optional[float] = None
    outputs: dict = field(default_factory=dict)
    demucs_applied: bool = False
    warnings: list = field(default_factory=list)
    cancelled: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path
