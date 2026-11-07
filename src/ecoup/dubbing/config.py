"""Configuration dataclasses for AI Dubbing module."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class AudioExtractConfig:
    sample_rate: int = 44100
    channels: int = 2

    def validate(self) -> None:
        if self.sample_rate not in (16000, 22050, 24000, 44100, 48000):
            raise ValueError(f"Invalid sample_rate: {self.sample_rate}. Expected one of 16000, 22050, 24000, 44100, 48000.")
        if self.channels not in (1, 2):
            raise ValueError(f"Invalid channels: {self.channels}. Expected 1 (mono) or 2 (stereo).")


@dataclass
class DemucsConfig:
    model_name: str = "htdemucs"
    two_stems: str = "vocals"
    device: str = "auto"  # "auto", "cuda", "cpu"
    mock: bool = False

    def validate(self) -> None:
        if not self.model_name:
            raise ValueError("model_name cannot be empty.")
        if self.two_stems not in ("vocals", "none"):
            raise ValueError(f"Unsupported two_stems value: {self.two_stems}")


@dataclass
class DubbingConfig:
    project_dir: Optional[str] = None
    audio: AudioExtractConfig = field(default_factory=AudioExtractConfig)
    demucs: DemucsConfig = field(default_factory=DemucsConfig)

    def validate(self) -> None:
        self.audio.validate()
        self.demucs.validate()

    def get_step_hash(self, step_name: str) -> str:
        """Compute MD5 hash for configuration of a specific pipeline step."""
        data: dict[str, Any] = {}
        if step_name == "S0":
            data = asdict(self.audio)
        elif step_name == "S1":
            data = {"audio": asdict(self.audio), "demucs": asdict(self.demucs)}
        else:
            data = asdict(self)
        raw = json.dumps(data, sort_keys=True)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()
