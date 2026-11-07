"""AI Dubbing module for ecoup."""

from .asr import AsrOptions, DubbingSegment, is_hallucination, transcribe_dialogue
from .audio import AudioProcessingError, NoAudioError, extract_original_hq, probe_audio_info, separate_stems
from .config import (
    AsrDubbingConfig,
    AudioExtractConfig,
    DemucsConfig,
    DiarizationDubbingConfig,
    DubbingConfig,
)
from .diarization import DiarizationOptions, SpeakerDiarizer, export_speaker_previews
from .manifest import ProjectManifest
from .pipeline import DubbingPipeline

__all__ = [
    "AudioExtractConfig",
    "DemucsConfig",
    "AsrDubbingConfig",
    "DiarizationDubbingConfig",
    "DubbingConfig",
    "ProjectManifest",
    "DubbingPipeline",
    "DubbingSegment",
    "SpeakerDiarizer",
    "DiarizationOptions",
    "AsrOptions",
    "AudioProcessingError",
    "NoAudioError",
    "extract_original_hq",
    "separate_stems",
    "probe_audio_info",
    "transcribe_dialogue",
    "export_speaker_previews",
    "is_hallucination",
]
