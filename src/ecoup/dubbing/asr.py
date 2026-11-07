"""Speech recognition (ASR) step for AI Dubbing module."""

from __future__ import annotations

import gc
import json
import re
import time
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np


@dataclass
class DubbingSegment:
    id: int
    start: float
    end: float
    text: str
    language: str = "en"
    confidence: float = 1.0
    speaker_id: Optional[str] = None
    is_overlap: bool = False

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 2)


_HALLUCINATION_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"^thanks for watching.*$",
        r"^thank you for watching.*$",
        r"^thank you!?\.?$",
        r"^thanks!?\.?$",
        r"^like and subscribe!?\.?$",
        r"^please subscribe!?\.?$",
        r"hãy subscribe cho kênh",
        r"đăng ký kênh",
    ]
]


def is_hallucination(text: str) -> bool:
    clean = re.sub(r"\s+", " ", text.strip()).lower()
    return any(p.search(clean) for p in _HALLUCINATION_PATTERNS)


def preload_cuda12_libs() -> None:
    """Load CUDA 12 libraries on Linux (useful when Colab ships CUDA 13)."""
    import os
    if os.name != "posix":
        return
    import ctypes
    import glob
    import importlib.util

    for pkg, patterns in (("nvidia.cublas", ["libcublasLt.so.12", "libcublas.so.12"]),
                          ("nvidia.cudnn", ["libcudnn.so.9"])):
        try:
            spec = importlib.util.find_spec(pkg)
        except Exception:
            spec = None
        if not spec or not spec.submodule_search_locations:
            continue
        lib_dir = os.path.join(list(spec.submodule_search_locations)[0], "lib")
        for pat in patterns:
            for f in glob.glob(os.path.join(lib_dir, pat)):
                try:
                    ctypes.CDLL(f, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample 1D audio array. Uses scipy.signal if available, otherwise numpy interpolation."""
    if orig_sr == target_sr or len(audio) == 0:
        return audio
    num_samples = int(len(audio) * target_sr / orig_sr)
    try:
        import scipy.signal
        return scipy.signal.resample(audio, num_samples).astype(np.float32)
    except Exception:
        # Fallback to pure numpy linear interpolation
        old_indices = np.linspace(0, len(audio) - 1, num=len(audio))
        new_indices = np.linspace(0, len(audio) - 1, num=num_samples)
        return np.interp(new_indices, old_indices, audio).astype(np.float32)


def load_wav_float32(path: str | Path, target_sr: int = 16000) -> np.ndarray:
    """Read a PCM WAV into a float32 mono numpy array in [-1.0, 1.0]."""
    path = Path(path)
    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    if sampwidth == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    if framerate != target_sr:
        audio = resample_audio(audio, framerate, target_sr)

    return audio


@dataclass
class AsrOptions:
    model_size: str = "base"
    beam_size: int = 5
    vad_filter: bool = True
    language: Optional[str] = None
    mock: bool = False


def transcribe_dialogue(
    audio_path: str | Path,
    options: Optional[AsrOptions] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> tuple[list[DubbingSegment], str, float]:
    """Transcribe dialogue audio into DubbingSegments. Returns (segments, language, confidence)."""
    options = options or AsrOptions()
    audio_path = Path(audio_path)

    if options.mock:
        mock_segs = [
            DubbingSegment(id=0, start=1.0, end=3.5, text="Hello world, testing dubbing.", language="en"),
            DubbingSegment(id=1, start=4.0, end=7.2, text="This is the second test sentence.", language="en"),
        ]
        return mock_segs, "en", 0.99

    preload_cuda12_libs()

    import torch
    from faster_whisper import WhisperModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    model = WhisperModel(options.model_size, device=device, compute_type=compute_type)
    audio_arr = load_wav_float32(audio_path, target_sr=16000)

    whisper_segments, info = model.transcribe(
        audio_arr,
        beam_size=options.beam_size,
        vad_filter=options.vad_filter,
        language=options.language,
    )

    detected_lang = info.language or "en"
    lang_prob = float(getattr(info, "language_probability", 1.0))

    segments: list[DubbingSegment] = []
    for s in whisper_segments:
        if should_cancel and should_cancel():
            break
        text = s.text.strip()
        if text and not is_hallucination(text):
            segments.append(
                DubbingSegment(
                    id=len(segments),
                    start=round(float(s.start), 2),
                    end=round(float(s.end), 2),
                    text=text,
                    language=detected_lang,
                    confidence=round(float(getattr(s, "avg_logprob", 0.0)), 3),
                )
            )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return segments, detected_lang, lang_prob
