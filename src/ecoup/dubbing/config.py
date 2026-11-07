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
    device: str = "auto"
    mock: bool = False

    def validate(self) -> None:
        if not self.model_name:
            raise ValueError("model_name cannot be empty.")
        if self.two_stems not in ("vocals", "none"):
            raise ValueError(f"Unsupported two_stems value: {self.two_stems}")


@dataclass
class AsrDubbingConfig:
    model_size: str = "base"
    beam_size: int = 5
    vad_filter: bool = True
    language: Optional[str] = None
    mock: bool = False

    def validate(self) -> None:
        valid_sizes = ("tiny", "base", "small", "medium", "large-v3")
        if self.model_size not in valid_sizes:
            raise ValueError(f"Invalid model_size: {self.model_size}. Expected one of {valid_sizes}")


@dataclass
class DiarizationDubbingConfig:
    num_speakers: Optional[int] = None
    min_speakers: int = 1
    max_speakers: int = 10
    distance_threshold: float = 0.45
    mock: bool = False

    def validate(self) -> None:
        if self.num_speakers is not None and self.num_speakers < 1:
            raise ValueError("num_speakers must be >= 1.")
        if not (0.0 < self.distance_threshold < 2.0):
            raise ValueError("distance_threshold must be between 0.0 and 2.0")


@dataclass
class TranslationDubbingConfig:
    model: str = "gemini-2.5-flash"
    syllables_per_second: float = 3.8
    max_rewrite_attempts: int = 2
    temperature: float = 0.3
    mock: bool = False

    def validate(self) -> None:
        if not self.model:
            raise ValueError("model cannot be empty.")
        if self.syllables_per_second <= 0:
            raise ValueError("syllables_per_second must be > 0.")


@dataclass
class DubbingConfig:
    project_dir: Optional[str] = None
    audio: AudioExtractConfig = field(default_factory=AudioExtractConfig)
    demucs: DemucsConfig = field(default_factory=DemucsConfig)
    asr: AsrDubbingConfig = field(default_factory=AsrDubbingConfig)
    diarization: DiarizationDubbingConfig = field(default_factory=DiarizationDubbingConfig)
    translation: TranslationDubbingConfig = field(default_factory=TranslationDubbingConfig)

    def validate(self) -> None:
        self.audio.validate()
        self.demucs.validate()
        self.asr.validate()
        self.diarization.validate()
        self.translation.validate()

    def get_step_hash(self, step_name: str) -> str:
        """Compute MD5 hash for configuration of a specific pipeline step."""
        data: dict[str, Any] = {}
        if step_name == "S0":
            data = asdict(self.audio)
        elif step_name == "S1":
            data = {"audio": asdict(self.audio), "demucs": asdict(self.demucs)}
        elif step_name == "S2":
            data = asdict(self.asr)
        elif step_name == "S3":
            data = asdict(self.diarization)
        elif step_name == "S4":
            data = {"diarization": asdict(self.diarization)}
        elif step_name == "S5":
            data = asdict(self.translation)
        else:
            data = asdict(self)
        raw = json.dumps(data, sort_keys=True)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()
