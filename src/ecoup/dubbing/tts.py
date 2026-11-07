"""Voice generation backends (S6) with Multi-speaker support."""

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


def apply_pitch_shift(audio: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """Pitch shift audio by semitones without changing duration via FFmpeg."""
    if abs(semitones) < 0.1 or len(audio) == 0:
        return audio

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return audio

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        in_p = tf.name
    out_p = in_p.replace(".wav", "_shifted.wav")

    try:
        _write_wav_pcm16(Path(in_p), audio, sr)
        # asetrate changes pitch and speed, then atempo restores speed
        scale = 2.0 ** (semitones / 12.0)
        new_sr = int(sr * scale)
        tempo = 1.0 / scale
        tempo_filter = f"asetrate={new_sr},atempo={tempo:.4f},aresample={sr}"

        cmd = [ffmpeg, "-y", "-v", "error", "-i", in_p, "-af", tempo_filter, "-vn", "-c:a", "pcm_s16le", out_p]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode == 0 and os.path.exists(out_p):
            with wave.open(out_p, "rb") as wf:
                raw = wf.readframes(wf.getnframes())
                return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        return audio
    finally:
        for p in (in_p, out_p):
            if os.path.exists(p):
                try:
                    os.unlink(p)
                except Exception:
                    pass


class VoiceBackend(Protocol):
    def synthesize(
        self,
        text: str,
        ref_audio_path: Optional[str | Path] = None,
        speed: float = 1.0,
        target_duration: Optional[float] = None,
        speaker_id: str = "spk_0",
        ref_text: Optional[str] = None,
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
        speaker_id: str = "spk_0",
        ref_text: Optional[str] = None,
    ) -> tuple[np.ndarray, int]:
        sr = 24000
        dur = target_duration if target_duration and target_duration > 0.5 else 2.0
        dur = dur / max(0.5, speed)
        samples = int(dur * sr)

        # Vary pitch per speaker: spk_0=200Hz, spk_1=320Hz, spk_2=160Hz
        freq_map = {"spk_0": 200, "spk_1": 320, "spk_2": 160}
        base_freq = freq_map.get(speaker_id, 240)

        t = np.linspace(0, dur, samples, endpoint=False)
        audio = (0.2 * np.sin(2 * np.pi * base_freq * t)).astype(np.float32)
        return audio, sr


class GTTSVoiceBackend:
    def __init__(self, lang: str = "vi"):
        self.lang = lang
        # Semi-tone offsets to give each character a unique timbre
        self.speaker_pitch_offsets = {
            "spk_0": 0.0,    # Normal
            "spk_1": 3.0,    # Higher pitch (female/young)
            "spk_2": -3.5,   # Deeper pitch (older/male)
            "spk_3": 1.5,    # Slightly higher
            "spk_4": -1.8,   # Slightly lower
        }

    def synthesize(
        self,
        text: str,
        ref_audio_path: Optional[str | Path] = None,
        speed: float = 1.0,
        target_duration: Optional[float] = None,
        speaker_id: str = "spk_0",
        ref_text: Optional[str] = None,
    ) -> tuple[np.ndarray, int]:
        from gtts import gTTS
        import io

        tts = gTTS(text=text, lang=self.lang)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)

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

            # Apply character-specific pitch shift if multiple speakers
            shift = self.speaker_pitch_offsets.get(speaker_id, 0.0)
            if abs(shift) > 0.1:
                audio = apply_pitch_shift(audio, sr, shift)

            return audio, sr
        finally:
            for p in (temp_mp3, temp_wav):
                if os.path.exists(p):
                    try:
                        os.unlink(p)
                    except Exception:
                        pass


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
        speaker_id: str = "spk_0",
        ref_text: Optional[str] = None,
    ) -> tuple[np.ndarray, int]:
        self._load()
        from f5_tts.infer.utils_infer import infer_process

        ref_audio = str(ref_audio_path) if ref_audio_path and os.path.exists(str(ref_audio_path)) else ""
        resolved_ref_text = ref_text or ""

        wav, sr, _ = infer_process(
            ref_audio=ref_audio,
            ref_text=resolved_ref_text,
            gen_text=text,
            model_obj=self._model,
            vocoder=self._vocoder,
            mel_spec_type="vocos",
            speed=speed,
            fix_duration=target_duration,
            device=self.device,
        )
        return wav.astype(np.float32), sr


def _preload_cuda_libs_if_needed() -> None:
    if os.name != "posix":
        return
    import ctypes
    import glob
    import importlib.util
    for pkg, patterns in (("nvidia.cublas", ["libcublasLt.so.12", "libcublas.so.12"]),
                          ("nvidia.cudnn", ["libcudnn.so.9"])):
        try:
            spec = importlib.util.find_spec(pkg)
            if spec and spec.submodule_search_locations:
                lib_dir = os.path.join(list(spec.submodule_search_locations)[0], "lib")
                for pat in patterns:
                    for f in glob.glob(os.path.join(lib_dir, pat)):
                        try:
                            ctypes.CDLL(f, mode=ctypes.RTLD_GLOBAL)
                        except Exception:
                            pass
        except Exception:
            pass


class OpenVoiceBackend:
    """Zero-shot cross-lingual Voice Cloning using OpenVoice v2 Tone Color Converter."""

    def __init__(self, device: str = "auto", base_lang: str = "vi"):
        self.device = device
        self.base_lang = base_lang
        self._converter = None
        self._base_tts = GTTSVoiceBackend(lang=base_lang)
        self._speaker_target_se: dict[str, Any] = {}

    def _load(self) -> None:
        if self._converter is not None:
            return
        import torch
        from huggingface_hub import hf_hub_download
        from openvoice.api import ToneColorConverter

        _preload_cuda_libs_if_needed()

        dev = self.device
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"

        ckpt_path = hf_hub_download(repo_id="myshell-ai/OpenVoiceV2", filename="converter/checkpoint.pth")
        config_path = hf_hub_download(repo_id="myshell-ai/OpenVoiceV2", filename="converter/config.json")
        self._converter = ToneColorConverter(config_path, device=dev)
        self._converter.load_ckpt(ckpt_path)
        self.device = dev

    def synthesize(
        self,
        text: str,
        ref_audio_path: Optional[str | Path] = None,
        speed: float = 1.0,
        target_duration: Optional[float] = None,
        speaker_id: str = "spk_0",
        ref_text: Optional[str] = None,
    ) -> tuple[np.ndarray, int]:
        base_audio, sr = self._base_tts.synthesize(
            text=text,
            speed=speed,
            speaker_id=speaker_id,
        )
        if not ref_audio_path or not os.path.exists(str(ref_audio_path)):
            return base_audio, sr

        try:
            self._load()
            from openvoice import se_extractor
            import tempfile
            import soundfile as sf

            ref_str = str(ref_audio_path)
            if speaker_id not in self._speaker_target_se:
                target_se, _ = se_extractor.get_se(ref_str, self._converter, target_dir="/tmp/ov_se", vad=False)
                self._speaker_target_se[speaker_id] = target_se
            target_se = self._speaker_target_se[speaker_id]

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_base:
                base_wav = f_base.name
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_out:
                cloned_wav = f_out.name

            try:
                _write_wav_pcm16(Path(base_wav), base_audio, sr)
                source_se, _ = se_extractor.get_se(base_wav, self._converter, target_dir="/tmp/ov_se", vad=False)
                self._converter.convert(
                    audio_src_path=base_wav,
                    src_se=source_se,
                    tgt_se=target_se,
                    output_path=cloned_wav,
                    message="@Antigravity",
                )
                cloned_audio, cloned_sr = sf.read(cloned_wav, dtype="float32")
                if cloned_audio.ndim > 1:
                    cloned_audio = cloned_audio.mean(axis=1)
                return cloned_audio, cloned_sr
            finally:
                for p in (base_wav, cloned_wav):
                    if os.path.exists(p):
                        try:
                            os.unlink(p)
                        except Exception:
                            pass
        except Exception:
            return base_audio, sr

