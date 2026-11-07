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
            raise ValueError(f"Invalid sample_rate: {self.sample_rate}.")
        if self.channels not in (1, 2):
            raise ValueError(f"Invalid channels: {self.channels}.")


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
            raise ValueError(f"Invalid model_size: {self.model_size}.")


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
class TtsDubbingConfig:
    backend: str = "openvoice"  # "openvoice", "f5", "gtts", "mock"
    default_speed: float = 1.0
    max_speedup: float = 1.35
    mock: bool = False

    def validate(self) -> None:
        valid_backends = ("openvoice", "clone", "f5", "gtts", "mock")
        if self.backend not in valid_backends:
            raise ValueError(f"Unsupported TTS backend: {self.backend}")


@dataclass
class MixingDubbingConfig:
    ducking_db: float = -8.0
    target_sr: int = 44100

    def validate(self) -> None:
        if self.ducking_db > 0:
            raise ValueError("ducking_db must be negative (e.g. -8.0 dB).")


@dataclass
class DubbingConfig:
    project_dir: Optional[str] = None
    audio: AudioExtractConfig = field(default_factory=AudioExtractConfig)
    demucs: DemucsConfig = field(default_factory=DemucsConfig)
    asr: AsrDubbingConfig = field(default_factory=AsrDubbingConfig)
    diarization: DiarizationDubbingConfig = field(default_factory=DiarizationDubbingConfig)
    translation: TranslationDubbingConfig = field(default_factory=TranslationDubbingConfig)
    tts: TtsDubbingConfig = field(default_factory=TtsDubbingConfig)
    mixing: MixingDubbingConfig = field(default_factory=MixingDubbingConfig)

    def validate(self) -> None:
        self.audio.validate()
        self.demucs.validate()
        self.asr.validate()
        self.diarization.validate()
        self.translation.validate()
        self.tts.validate()
        self.mixing.validate()

    def get_step_hash(self, step_name: str) -> str:
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
        elif step_name == "S6":
            data = asdict(self.tts)
        elif step_name == "S7":
            data = {"speedup": self.tts.max_speedup}
        elif step_name == "S8":
            data = asdict(self.mixing)
        elif step_name == "S9":
            data = {"mux": "stream_copy"}
        else:
            data = asdict(self)
        raw = json.dumps(data, sort_keys=True)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()
