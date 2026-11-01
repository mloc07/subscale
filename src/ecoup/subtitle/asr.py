"""Speech recognition with faster-whisper (optional dependency)."""

from __future__ import annotations

import gc
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

from .config import AsrConfig
from .segments import Segment


@dataclass
class AsrResult:
    segments: list[Segment]
    language: Optional[str]
    language_probability: float = 0.0
    duration_s: float = 0.0
    elapsed_s: float = 0.0
    peak_vram_gb: Optional[float] = None


def _preload_cuda12_libs() -> None:
    """ctranslate2 needs CUDA 12 cuBLAS/cuDNN. When the system CUDA differs (e.g. Colab ships CUDA 13),
    load the pip wheels ``nvidia-cublas-cu12`` / ``nvidia-cudnn-cu12`` by absolute path (Linux only)."""
    import ctypes
    import glob
    import importlib.util
    import os

    if os.name != "posix":
        return
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


def _gpu_used_gb() -> Optional[float]:
    """GPU memory currently used (GiB) according to nvidia-smi; None if unavailable."""
    import shutil
    import subprocess

    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10, check=False).stdout.strip().splitlines()
        return float(out[0]) / 1024.0 if out else None
    except Exception:
        return None


_HALLUCINATION_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"^thanks for watching!?\.?$",
        r"^thank you for watching!?\.?$",
        r"^thanks for watching this video!?\.?$",
        r"^thank you!?\.?$",
        r"^thanks!?\.?$",
        r"^thank you very much!?\.?$",
        r"hãy subscribe cho kênh",
        r"subscribe for more",
        r"đăng ký kênh để không bỏ lỡ",
        r"^like and subscribe!?\.?$",
        r"^please subscribe!?\.?$",
        r"kênh la la school",
    ]
]


def _is_hallucination(text: str) -> bool:
    clean = re.sub(r"\s+", " ", text.strip()).lower()
    return any(p.search(clean) for p in _HALLUCINATION_PATTERNS)


def _load_wav_float32(path):
    """Read a 16-bit PCM mono WAV into a float32 numpy array in [-1, 1]."""
    import wave

    import numpy as np

    with wave.open(str(path), "rb") as wf:
        if wf.getsampwidth() != 2:
            raise ValueError("Expected 16-bit PCM WAV.")
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


class AsrBackend(Protocol):
    def transcribe(self, wav_path: str, cfg: AsrConfig,
                   on_segment: Optional[Callable[[float], None]] = None,
                   should_cancel: Optional[Callable[[], bool]] = None) -> AsrResult: ...


class FasterWhisperBackend:
    """Loads the model, transcribes, then releases it (GPU memory is freed afterwards)."""

    def transcribe(self, wav_path, cfg, on_segment=None, should_cancel=None) -> AsrResult:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "faster-whisper is not installed. Install the optional extra: pip install 'ecoup[subtitle]'"
            ) from exc

        device = cfg.device
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        compute = cfg.compute_type
        if compute == "auto":
            compute = "float16" if device == "cuda" else "int8"

        _preload_cuda12_libs()
        base_used = _gpu_used_gb() if device == "cuda" else None
        t0 = time.time()
        model = WhisperModel(cfg.model, device=device, compute_type=compute)
        peak = None
        # Configure VAD presets based on audio_mode
        vad_filter = cfg.vad_filter
        vad_params = {}
        if cfg.audio_mode == "off" or cfg.use_demucs:
            # Demucs isolates clean vocals; disabling Silero VAD avoids clipping singing vibrato
            vad_filter = False
        elif cfg.audio_mode == "music":
            vad_filter = True
            vad_params = dict(
                threshold=cfg.vad_threshold if cfg.vad_threshold <= 0.35 else 0.25,
                min_silence_duration_ms=cfg.vad_min_silence_ms,
                speech_pad_ms=cfg.speech_pad_ms,
            )
        else:  # "speech"
            vad_filter = cfg.vad_filter
            vad_params = dict(
                threshold=0.50,
                min_silence_duration_ms=500,
                speech_pad_ms=250,
            )

        kwargs = dict(
            beam_size=cfg.beam_size,
            language=cfg.language,
            vad_filter=vad_filter,
            condition_on_previous_text=cfg.condition_on_previous_text,
            no_speech_threshold=cfg.no_speech_threshold,
        )
        if cfg.initial_prompt:
            kwargs["initial_prompt"] = cfg.initial_prompt
        if vad_filter and vad_params:
            kwargs["vad_parameters"] = vad_params

        audio_arr = _load_wav_float32(wav_path)
        sr = 16000
        total_sec = len(audio_arr) / sr
        segs: list[Segment] = []

        try:
            # Monolithic transcribe is default and most reliable (no artificial cuts at multiples of 25s)
            if not cfg.multilingual_chunking or cfg.language is not None or total_sec <= 30.0:
                seg_iter, info = model.transcribe(audio_arr, **kwargs)
                det_lang = info.language
                if cfg.language_whitelist and det_lang not in cfg.language_whitelist:
                    det_lang = cfg.language_whitelist[0]
                for s in seg_iter:
                    if should_cancel and should_cancel():
                        break
                    s_text = s.text.strip()
                    if s_text and not _is_hallucination(s_text):
                        segs.append(Segment(start=float(s.start), end=float(s.end), text=s_text, language=det_lang))
                    if on_segment:
                        on_segment(float(s.end))
                detected_lang = det_lang
                lang_prob = float(getattr(info, "language_probability", 0.0))
            else:
                # Dynamic multilingual chunking guided by VAD silence pauses (NOT rigid 25s intervals)
                try:
                    from faster_whisper.vad import get_speech_timestamps, VadOptions
                    v_opts = VadOptions(
                        threshold=vad_params.get("threshold", 0.35) if vad_params else 0.35,
                        min_silence_duration_ms=600,
                        speech_pad_ms=400,
                    )
                    speech_blocks = get_speech_timestamps(audio_arr, vad_options=v_opts, sampling_rate=sr)
                except Exception:
                    speech_blocks = []

                if not speech_blocks:
                    # Fallback to monolithic if VAD finds no distinct blocks
                    seg_iter, info = model.transcribe(audio_arr, **kwargs)
                    det_lang = info.language
                    if cfg.language_whitelist and det_lang not in cfg.language_whitelist:
                        det_lang = cfg.language_whitelist[0]
                    for s in seg_iter:
                        if should_cancel and should_cancel():
                            break
                        s_text = s.text.strip()
                        if s_text and not _is_hallucination(s_text):
                            segs.append(Segment(start=float(s.start), end=float(s.end), text=s_text, language=det_lang))
                    detected_lang, lang_prob = det_lang, float(getattr(info, "language_probability", 0.0))
                else:
                    # Group speech blocks into natural chunks (15s to 35s) at natural speech pauses
                    chunks: list[tuple[float, float]] = []
                    c_start = max(0.0, float(speech_blocks[0]["start"]) / sr - 0.5)
                    c_end = float(speech_blocks[0]["end"]) / sr + 0.5
                    for blk in speech_blocks[1:]:
                        b_s = float(blk["start"]) / sr
                        b_e = float(blk["end"]) / sr
                        if (b_e - c_start) > 30.0 or (b_s - c_end) > 1.2:
                            chunks.append((c_start, min(total_sec, c_end + 0.5)))
                            c_start = max(0.0, b_s - 0.5)
                        c_end = b_e
                    chunks.append((c_start, min(total_sec, c_end + 0.5)))

                    detected_langs: list[tuple[str, float]] = []
                    for c_start, c_end in chunks:
                        if should_cancel and should_cancel():
                            break
                        c_slice = audio_arr[int(c_start * sr) : int(c_end * sr)]
                        c_iter, c_info = model.transcribe(c_slice, **kwargs)
                        chunk_lang = c_info.language if c_info else None
                        if cfg.language_whitelist and chunk_lang and chunk_lang not in cfg.language_whitelist:
                            chunk_lang = cfg.language_whitelist[0]
                        if chunk_lang:
                            detected_langs.append((chunk_lang, float(getattr(c_info, "language_probability", 0.0))))
                        for s in c_iter:
                            if should_cancel and should_cancel():
                                break
                            s_text = s.text.strip()
                            if s_text and not _is_hallucination(s_text):
                                segs.append(Segment(
                                    start=round(c_start + float(s.start), 3),
                                    end=round(c_start + float(s.end), 3),
                                    text=s_text,
                                    language=chunk_lang,
                                ))
                        if on_segment:
                            on_segment(c_end)

                    if detected_langs:
                        from collections import Counter
                        counts = Counter([l[0] for l in detected_langs])
                        detected_lang = counts.most_common(1)[0][0]
                        lang_prob = max((p for l, p in detected_langs if l == detected_lang), default=0.5)
                    else:
                        detected_lang, lang_prob = "auto", 0.0

            if device == "cuda":
                # ctranslate2 allocates outside PyTorch, so torch.cuda counters read 0: ask the driver.
                used = _gpu_used_gb()
                peak = None if used is None or base_used is None else max(0.0, used - base_used)
            return AsrResult(
                segments=segs,
                language=detected_lang,
                language_probability=lang_prob,
                duration_s=total_sec,
                elapsed_s=time.time() - t0,
                peak_vram_gb=peak,
            )
        finally:
            del model
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
