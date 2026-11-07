"""Voice generation backends (S6)."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
import wave
from pathlib import Path
from typing import Optional, Protocol

import numpy as np

from .asr import resample_audio
from .diarization import _write_wav_pcm16


class VoiceBackend(Protocol):
    def synthesize(
        self,
        text: str,
        ref_audio_path: Optional[str | Path] = None,
        speed: float = 1.0,
        target_duration: Optional[float] = None,
    ) -> tuple[np.ndarray, int]:
        """Synthesize Vietnamese text into audio array and sample rate."""
        ...


class MockVoiceBackend:
    def synthesize(
        self,
        text: str,
        ref_audio_path: Optional[str | Path] = None,
        speed: float = 1.0,
        target_duration: Optional[float] = None,
    ) -> tuple[np.ndarray, int]:
        sr = 24000
        dur = target_duration if target_duration and target_duration > 0.5 else 2.0
        dur = dur / max(0.5, speed)
        samples = int(dur * sr)
        # Generate synthetic sine wave with speech-like envelope
        t = np.linspace(0, dur, samples, endpoint=False)
        audio = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        return audio, sr


class GTTSVoiceBackend:
    def __init__(self, lang: str = "vi"):
        self.lang = lang

    def synthesize(
        self,
        text: str,
        ref_audio_path: Optional[str | Path] = None,
        speed: float = 1.0,
        target_duration: Optional[float] = None,
    ) -> tuple[np.ndarray, int]:
        from gtts import gTTS
        import io
        import soundfile as sf

        tts = gTTS(text=text, lang=self.lang)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)

        # Read mp3 stream via soundfile / ffmpeg
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tf:
            tf.write(fp.read())
            temp_mp3 = tf.name

        temp_wav = temp_mp3.replace(".mp3", ".wav")
        try:
            cmd = ["ffmpeg", "-y", "-v", "error", "-i", temp_mp3, "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", temp_wav]
            subprocess.run(cmd, check=True)
            with wave.open(temp_wav, "rb") as wf:
                raw = wf.readframes(wf.getnframes())
                sr = wf.getframerate()
                audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        finally:
            for p in (temp_mp3, temp_wav):
                if os.path.exists(p):
                    try:
                        os.unlink(p)
                    except Exception:
                        pass

        return audio, sr


class F5VoiceBackend:
    def __init__(self, repo_id: str = "danhtran2mind/Vi-F5-TTS", device: str = "auto"):
        self.repo_id = repo_id
        self.device = device
        self._model = None
        self._vocoder = None
        self._vocab_path = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from f5_tts.infer.utils_infer import load_model, load_vocoder
        from f5_tts.model import DiT
        from huggingface_hub import hf_hub_download

        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        ckpt_path = hf_hub_download(repo_id=self.repo_id, filename="Vi_F5_TTS_ckpts/model_last.pt")
        vocab_path = hf_hub_download(repo_id=self.repo_id, filename="vocab.txt")
        self._vocab_path = vocab_path

        self._vocoder = load_vocoder(vocoder_name="vocos", device=device)
        model_cls = DiT
        model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
        self._model = load_model(
            model_cls,
            model_cfg,
            ckpt_path,
            mel_spec_type="vocos",
            vocab_file=vocab_path,
            device=device,
        )
        self.device = device

    def synthesize(
        self,
        text: str,
        ref_audio_path: Optional[str | Path] = None,
        speed: float = 1.0,
        target_duration: Optional[float] = None,
    ) -> tuple[np.ndarray, int]:
        self._load()
        from f5_tts.infer.utils_infer import infer_process

        ref_audio = str(ref_audio_path) if ref_audio_path else ""
        ref_text = ""  # If zero-shot without ref text, F5 can run unconditional or with prompt

        wav, sr, _ = infer_process(
            ref_audio=ref_audio,
            ref_text=ref_text,
            gen_text=text,
            model_obj=self._model,
            vocoder=self._vocoder,
            mel_spec_type="vocos",
            speed=speed,
            fix_duration=target_duration,
            device=self.device,
        )
        return wav.astype(np.float32), sr
