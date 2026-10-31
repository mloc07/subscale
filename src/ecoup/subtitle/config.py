"""Configuration for the automatic subtitle module (ASR -> translate -> burn-in).

Kept standalone from ``ecoup.config`` so the module can be renamed/moved easily.
API keys are NEVER stored here: they are read from environment variables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AsrConfig:
    model: str = "large-v3"
    device: str = "auto"              # "auto" | "cuda" | "cpu"
    compute_type: str = "auto"        # "auto" | "float16" | "int8_float16" | "int8" | "float32"
    language: Optional[str] = None    # None = auto-detect; otherwise ISO code, e.g. "en", "zh", "ja", "ko"
    audio_mode: str = "speech"        # "speech" (standard VAD for vlogs) | "music" (optimized for songs/beats) | "off" (no VAD)
    multilingual_chunking: bool = False # Off by default; when on, uses silence-based chunking
    vad_filter: bool = True           # Silero VAD: reduces hallucinated text in silence/music
    vad_min_silence_ms: int = 800
    vad_threshold: float = 0.25
    speech_pad_ms: int = 400
    no_speech_threshold: float = 0.6
    beam_size: int = 5
    condition_on_previous_text: bool = False  # False reduces repetition loops
    initial_prompt: Optional[str] = None      # Vocabulary / lyrics / proper nouns hint
    language_whitelist: Optional[list[str]] = None # Restrict allowed languages when auto-detecting
    use_demucs: bool = False                  # Isolate vocals using Demucs before Whisper ASR

    def validate(self) -> None:
        if not self.model:
            raise ValueError("subtitle.asr.model must be non-empty.")
        if self.device not in ("auto", "cuda", "cpu"):
            raise ValueError(f"subtitle.asr.device must be auto/cuda/cpu, got '{self.device}'.")
        if self.audio_mode not in ("music", "speech", "off"):
            raise ValueError(f"subtitle.asr.audio_mode must be 'music', 'speech', or 'off', got '{self.audio_mode}'.")
        if self.beam_size < 1:
            raise ValueError("subtitle.asr.beam_size must be >= 1.")
        if self.vad_min_silence_ms < 0:
            raise ValueError("subtitle.asr.vad_min_silence_ms must be >= 0.")


@dataclass
class SegmentConfig:
    max_chars_per_line: int = 42
    max_lines: int = 2
    min_duration: float = 0.8         # seconds; shorter segments are merged with a neighbour
    max_duration: float = 7.0         # seconds; longer segments are split
    max_chars_per_second: float = 20.0
    merge_gap: float = 0.3            # only merge short segments separated by <= this gap

    def validate(self) -> None:
        if self.max_chars_per_line < 8:
            raise ValueError("subtitle.segment.max_chars_per_line must be >= 8.")
        if self.max_lines < 1:
            raise ValueError("subtitle.segment.max_lines must be >= 1.")
        if self.min_duration <= 0 or self.max_duration <= self.min_duration:
            raise ValueError("subtitle.segment: need 0 < min_duration < max_duration.")
        if self.max_chars_per_second <= 0:
            raise ValueError("subtitle.segment.max_chars_per_second must be > 0.")


@dataclass
class TranslateConfig:
    backend: str = "gemini"
    model: str = "gemini-3.1-flash-lite"   # verified via models.list (see docs/subtitle_notes.md)
    fallback_models: list = field(default_factory=lambda: ["gemini-3.5-flash-lite", "gemini-2.5-flash-lite"])
    target_language: str = "vi"
    batch_size: int = 25               # segments per request
    context_segments: int = 3          # previous/next segments given as read-only context
    temperature: float = 0.3
    timeout_s: float = 120.0
    max_retries: int = 4
    api_key_env: str = "GEMINI_API_KEY"
    safety_threshold: str = "default"  # "default" | "block_none" -- only the API's own legitimate setting
    context_note: str = ""             # free-form note from the user (genre, speakers, relationships)
    glossary: dict = field(default_factory=dict)  # {"source term": "bản dịch"}
    cache_dir: Optional[str] = None

    def validate(self) -> None:
        if self.backend not in ("gemini",):
            raise ValueError(f"subtitle.translate.backend unsupported: '{self.backend}'.")
        if self.batch_size < 1:
            raise ValueError("subtitle.translate.batch_size must be >= 1.")
        if self.context_segments < 0:
            raise ValueError("subtitle.translate.context_segments must be >= 0.")
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError("subtitle.translate.temperature must be in [0, 2].")
        if self.max_retries < 0:
            raise ValueError("subtitle.translate.max_retries must be >= 0.")
        if self.safety_threshold not in ("default", "block_none"):
            raise ValueError("subtitle.translate.safety_threshold must be 'default' or 'block_none'.")


@dataclass
class BurnConfig:
    font_name: str = "Noto Sans"       # must contain Vietnamese diacritics
    fonts_dir: Optional[str] = None
    font_scale: float = 0.045          # font size = scale * video height
    outline_scale: float = 0.12        # outline = scale * font size
    margin_v_scale: float = 0.06       # bottom margin = scale * video height
    margin_h_scale: float = 0.04
    primary_colour: str = "&H00FFFFFF"
    outline_colour: str = "&H00000000"
    bold: bool = True
    show_original: bool = False        # also show source text above the translation (smaller)

    def validate(self) -> None:
        if not self.font_name:
            raise ValueError("subtitle.burn.font_name must be non-empty.")
        if not (0.01 <= self.font_scale <= 0.2):
            raise ValueError("subtitle.burn.font_scale must be in [0.01, 0.2].")


@dataclass
class SubtitleConfig:
    asr: AsrConfig = field(default_factory=AsrConfig)
    segment: SegmentConfig = field(default_factory=SegmentConfig)
    translate: TranslateConfig = field(default_factory=TranslateConfig)
    burn: BurnConfig = field(default_factory=BurnConfig)
    burn_in: bool = True               # False = only produce .srt files
    review_before_burn: bool = False   # reserved for a later iteration (preview/edit step)
    work_dir: Optional[str] = None

    def validate(self) -> None:
        self.asr.validate()
        self.segment.validate()
        self.translate.validate()
        self.burn.validate()
