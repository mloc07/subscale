"""AI Dubbing module for ecoup."""

from .asr import AsrOptions, DubbingSegment, is_hallucination, transcribe_dialogue
from .audio import AudioProcessingError, NoAudioError, extract_original_hq, probe_audio_info, separate_stems
from .config import (
    AsrDubbingConfig,
    AudioExtractConfig,
    DemucsConfig,
    DiarizationDubbingConfig,
    DubbingConfig,
    TranslationDubbingConfig,
)
from .diarization import DiarizationOptions, SpeakerDiarizer, export_speaker_previews
from .manifest import ProjectManifest
from .pipeline import DubbingPipeline
from .translate import (
    DubbingTranslator,
    TranslatedDubSegment,
    TranslationConfig,
    count_vietnamese_syllables,
)
from .voice_profile import VoiceProfile, build_voice_profiles, calculate_audio_quality

__all__ = [
    "AudioExtractConfig",
    "DemucsConfig",
    "AsrDubbingConfig",
    "DiarizationDubbingConfig",
    "TranslationDubbingConfig",
    "DubbingConfig",
    "ProjectManifest",
    "DubbingPipeline",
    "DubbingSegment",
    "TranslatedDubSegment",
    "VoiceProfile",
    "SpeakerDiarizer",
    "DubbingTranslator",
    "DiarizationOptions",
    "AsrOptions",
    "TranslationConfig",
    "AudioProcessingError",
    "NoAudioError",
    "extract_original_hq",
    "separate_stems",
    "probe_audio_info",
    "transcribe_dialogue",
    "export_speaker_previews",
    "build_voice_profiles",
    "calculate_audio_quality",
    "count_vietnamese_syllables",
    "is_hallucination",
]
