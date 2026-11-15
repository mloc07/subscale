"""AI Dubbing module for ecoup."""

from .asr import AsrOptions, DubbingSegment, is_hallucination, transcribe_dialogue
from .audio import AudioProcessingError, NoAudioError, extract_original_hq, probe_audio_info, separate_stems
from .config import (
    AsrDubbingConfig,
    AudioExtractConfig,
    DemucsConfig,
    DiarizationDubbingConfig,
    DubbingConfig,
    MixingDubbingConfig,
    TranslationDubbingConfig,
    TtsDubbingConfig,
)
from .diarization import DiarizationOptions, SpeakerDiarizer, export_speaker_previews
from .manifest import ProjectManifest
from .mixing import mix_audio_tracks
from .mux import export_srt_file, mux_video_with_audio
from .pipeline import DubbingPipeline
from .snapping import align_audio_to_target_duration
from .translate import (
    DubbingTranslator,
    TranslatedDubSegment,
    TranslationConfig,
    count_vietnamese_syllables,
)
from .tts import (
    F5VoiceBackend,
    GTTSVoiceBackend,
    MockVoiceBackend,
    OpenVoiceBackend,
    VoiceBackend,
    estimate_median_f0_hz,
    export_audio_mp3,
    synthesize_text_to_mp3,
)
from .voice_bank import (
    SPECIAL_ORIGINAL,
    SPECIAL_SELF,
    ClonedVoiceEntry,
    VoiceBankManager,
    clone_voices_from_video,
    suggest_speaker_mapping,
)
from .voice_profile import VoiceProfile, build_voice_profiles, calculate_audio_quality

__all__ = [
    "AudioExtractConfig",
    "DemucsConfig",
    "AsrDubbingConfig",
    "DiarizationDubbingConfig",
    "TranslationDubbingConfig",
    "TtsDubbingConfig",
    "MixingDubbingConfig",
    "DubbingConfig",
    "ProjectManifest",
    "DubbingPipeline",
    "DubbingSegment",
    "TranslatedDubSegment",
    "VoiceProfile",
    "ClonedVoiceEntry",
    "VoiceBankManager",
    "SPECIAL_SELF",
    "SPECIAL_ORIGINAL",
    "SpeakerDiarizer",
    "DubbingTranslator",
    "VoiceBackend",
    "MockVoiceBackend",
    "GTTSVoiceBackend",
    "F5VoiceBackend",
    "OpenVoiceBackend",
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
    "clone_voices_from_video",
    "suggest_speaker_mapping",
    "estimate_median_f0_hz",
    "export_audio_mp3",
    "synthesize_text_to_mp3",
    "count_vietnamese_syllables",
    "is_hallucination",
    "align_audio_to_target_duration",
    "mix_audio_tracks",
    "export_srt_file",
    "mux_video_with_audio",
]
