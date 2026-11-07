"""AI Dubbing module for ecoup."""

from .audio import AudioProcessingError, NoAudioError, extract_original_hq, probe_audio_info, separate_stems
from .config import AudioExtractConfig, DemucsConfig, DubbingConfig
from .manifest import ProjectManifest
from .pipeline import DubbingPipeline

__all__ = [
    "AudioExtractConfig",
    "DemucsConfig",
    "DubbingConfig",
    "ProjectManifest",
    "DubbingPipeline",
    "AudioProcessingError",
    "NoAudioError",
    "extract_original_hq",
    "separate_stems",
    "probe_audio_info",
]
