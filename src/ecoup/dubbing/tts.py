"""Voice generation backends (S6) with Multi-speaker support."""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import tempfile
import time
import wave
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

import numpy as np

from .asr import load_wav_float32, preload_cuda12_libs, resample_audio
from .diarization import _write_wav_pcm16
from .snapping import change_audio_speed_ffmpeg


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
    def __init__(self, lang: str = "vi", apply_speaker_offset: bool = True):
        self.lang = lang
        self.apply_speaker_offset = apply_speaker_offset
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

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tf:
            tf.write(fp.read())
            temp_mp3 = tf.name

        temp_wav = temp_mp3.replace(".mp3", ".wav")
        try:
            ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
            cmd = [ffmpeg, "-y", "-v", "error", "-i", temp_mp3, "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", temp_wav]
            subprocess.run(cmd, check=True)
            with wave.open(temp_wav, "rb") as wf:
                raw = wf.readframes(wf.getnframes())
                sr = wf.getframerate()
                audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

            # Apply character-specific pitch shift only when used standalone (not before OpenVoice)
            if self.apply_speaker_offset:
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


def estimate_median_f0_hz(
    audio: np.ndarray,
    sr: int,
    fmin: float = 65.0,
    fmax: float = 360.0,
) -> float:
    """Estimate median fundamental frequency (F0 in Hz) of voiced frames using autocorrelation."""
    if len(audio) < int(0.05 * sr) or sr <= 0:
        return 0.0

    # Downsample to <= 16kHz for fast pitch estimation
    work_sr = sr
    work = audio.astype(np.float32, copy=False)
    if work_sr > 16000:
        work = resample_audio(work, work_sr, 16000)
        work_sr = 16000

    peak_amp = float(np.max(np.abs(work)))
    if peak_amp < 1e-4:
        return 0.0

    frame_len = int(0.04 * work_sr)
    hop_len = int(0.02 * work_sr)
    min_lag = max(1, int(work_sr / fmax))
    max_lag = min(frame_len - 2, int(work_sr / fmin))
    if max_lag <= min_lag:
        return 0.0

    rms_thresh = max(0.008, peak_amp * 0.12)
    f0_values: list[float] = []

    for start in range(0, len(work) - frame_len + 1, hop_len):
        frame = work[start : start + frame_len]
        rms = float(np.sqrt(np.mean(frame * frame)))
        if rms < rms_thresh:
            continue

        frame = frame - float(np.mean(frame))
        energy = float(np.dot(frame, frame))
        if energy <= 1e-8:
            continue

        # Autocorrelation on positive lags via FFT
        n_fft = 1 << (2 * frame_len - 1).bit_length()
        spec = np.fft.rfft(frame, n=n_fft)
        acf = np.fft.irfft(spec * np.conj(spec), n=n_fft)[:frame_len] / energy

        segment = acf[min_lag : max_lag + 1]
        if len(segment) == 0:
            continue
        best_idx = int(np.argmax(segment))
        best_val = float(segment[best_idx])
        if best_val >= 0.42:
            lag = min_lag + best_idx
            f0_values.append(float(work_sr) / float(lag))

    if not f0_values:
        return 0.0
    return round(float(np.median(f0_values)), 2)


def compute_pitch_shift_semitones(
    base_audio: np.ndarray,
    base_sr: int,
    target_f0_hz: float,
    min_semitones: float = -6.0,
    max_semitones: float = 5.0,
) -> float:
    """Compute semitone shift to match base_audio F0 to target speaker F0."""
    if target_f0_hz <= 0.0:
        return 0.0
    base_f0 = estimate_median_f0_hz(base_audio, base_sr)
    if base_f0 <= 0.0:
        return 0.0
    semitones = 12.0 * math.log2(target_f0_hz / base_f0)
    semitones = max(min_semitones, min(max_semitones, semitones))
    return 0.0 if abs(semitones) < 0.6 else round(semitones, 2)


class OpenVoiceBackend:
    """Zero-shot cross-lingual Voice Cloning using OpenVoice v2 Tone Color Converter."""

    def __init__(
        self,
        device: str = "auto",
        base_lang: str = "vi",
        pitch_match_mode: str = "auto",
        pitch_shift_semitones: float = 0.0,
    ):
        self.device = device
        self.base_lang = base_lang
        self.pitch_match_mode = pitch_match_mode
        self.pitch_shift_semitones = pitch_shift_semitones
        self._converter = None
        self._base_tts = GTTSVoiceBackend(lang=base_lang, apply_speaker_offset=False)
        self._speaker_target_se: dict[tuple[str, str], Any] = {}
        self._ref_f0_cache: dict[str, float] = {}
        self._se_temp_dir = Path(tempfile.gettempdir()) / "ecoup_ov_se"
        self._se_temp_dir.mkdir(parents=True, exist_ok=True)

    def _load(self) -> None:
        if self._converter is not None:
            return
        import torch
        from huggingface_hub import hf_hub_download
        from openvoice.api import ToneColorConverter

        preload_cuda12_libs()

        dev = self.device
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"

        ckpt_path = hf_hub_download(repo_id="myshell-ai/OpenVoiceV2", filename="converter/checkpoint.pth")
        config_path = hf_hub_download(repo_id="myshell-ai/OpenVoiceV2", filename="converter/config.json")
        self._converter = ToneColorConverter(config_path, device=dev)
        self._converter.load_ckpt(ckpt_path)
        self.device = dev

    def _extract_se_safe(self, wav_path: str | Path) -> Any:
        """Extract OpenVoice tone-color embedding, falling back directly to converter if Whisper yields 0 segments."""
        from openvoice import se_extractor

        wav_str = str(Path(wav_path).resolve())
        try:
            target_se, _ = se_extractor.get_se(
                wav_str,
                self._converter,
                target_dir=str(self._se_temp_dir),
                vad=False,
            )
            return target_se
        except Exception:
            # Direct ToneColorConverter extraction handles short/non-lexical clips cleanly
            return self._converter.extract_se([wav_str], se_save_path=None)

    def extract_speaker_embedding(
        self,
        ref_audio_path: str | Path,
        save_pt_path: Optional[str | Path] = None,
    ) -> Any:
        """Extract and optionally persist an OpenVoice speaker embedding (.pt)."""
        self._load()
        import torch

        target_se = self._extract_se_safe(ref_audio_path)
        if save_pt_path is not None:
            save_p = Path(save_pt_path)
            save_p.parent.mkdir(parents=True, exist_ok=True)
            torch.save(target_se.detach().cpu(), str(save_p))
        return target_se

    def _get_target_f0(self, ref_path: Path) -> float:
        resolved = str(ref_path.resolve())
        if resolved in self._ref_f0_cache:
            return self._ref_f0_cache[resolved]

        f0 = 0.0
        meta_json = ref_path.with_name("clone_meta.json")
        if meta_json.exists():
            try:
                import json
                meta = json.loads(meta_json.read_text(encoding="utf-8"))
                f0 = float(meta.get("estimated_f0_hz", 0.0) or 0.0)
            except Exception:
                f0 = 0.0

        if f0 <= 0.0 and ref_path.exists():
            try:
                ref_audio = load_wav_float32(ref_path, target_sr=16000)
                f0 = estimate_median_f0_hz(ref_audio, 16000)
            except Exception:
                f0 = 0.0

        self._ref_f0_cache[resolved] = f0
        return f0

    def _get_target_se(self, ref_path: Path, speaker_id: str) -> Any:
        import torch

        resolved = str(ref_path.resolve())
        cache_key = (speaker_id, resolved)
        if cache_key in self._speaker_target_se:
            return self._speaker_target_se[cache_key]

        emb_pt = ref_path.with_name("embedding.pt")
        if emb_pt.exists():
            try:
                target_se = torch.load(str(emb_pt), map_location=self.device, weights_only=True)
            except TypeError:
                target_se = torch.load(str(emb_pt), map_location=self.device)
            target_se = target_se.to(self.device)
            self._speaker_target_se[cache_key] = target_se
            return target_se

        target_se = self.extract_speaker_embedding(ref_path)
        self._speaker_target_se[cache_key] = target_se
        return target_se

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

        ref_path = Path(ref_audio_path)

        # Optional F0 alignment before tone conversion for natural male/female register
        shift_semitones = 0.0
        if self.pitch_match_mode == "manual":
            shift_semitones = self.pitch_shift_semitones
        elif self.pitch_match_mode == "auto":
            target_f0 = self._get_target_f0(ref_path)
            shift_semitones = compute_pitch_shift_semitones(base_audio, sr, target_f0)

        if abs(shift_semitones) >= 0.5:
            base_audio = apply_pitch_shift(base_audio, sr, shift_semitones)

        try:
            self._load()
            import soundfile as sf

            target_se = self._get_target_se(ref_path, speaker_id)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_base:
                base_wav = f_base.name
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_out:
                cloned_wav = f_out.name

            try:
                _write_wav_pcm16(Path(base_wav), base_audio, sr)
                source_se = self._extract_se_safe(base_wav)
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


def split_tts_chunks(text: str, max_chars: int = 180) -> list[tuple[str, float]]:
    """Split long TTS text into sentence/clause chunks with natural pause durations."""
    clean = text.strip()
    if not clean:
        return []

    chunks: list[tuple[str, float]] = []
    paragraphs = [p.strip() for p in re.split(r"\n+", clean) if p.strip()]
    for p_idx, para in enumerate(paragraphs):
        sentences = re.split(r"(?<=[\.\!\?\…])\s+", para)
        for s_idx, sent in enumerate(sentences):
            sent = sent.strip()
            if not sent:
                continue
            is_last_sent = s_idx == len(sentences) - 1
            end_pause = 0.35 if (is_last_sent and p_idx < len(paragraphs) - 1) else 0.22

            if len(sent) <= max_chars:
                chunks.append((sent, end_pause))
                continue

            # Split long sentence on commas/semicolons or spaces
            parts = re.split(r"(?<=[,;:])\s+", sent)
            buf = ""
            for part_idx, part in enumerate(parts):
                candidate = f"{buf} {part}".strip() if buf else part
                if len(candidate) <= max_chars:
                    buf = candidate
                else:
                    if buf:
                        chunks.append((buf, 0.12))
                        buf = ""
                    if len(part) <= max_chars:
                        buf = part
                    else:
                        words = part.split()
                        for w in words:
                            cand_w = f"{buf} {w}".strip() if buf else w
                            if len(cand_w) <= max_chars:
                                buf = cand_w
                            else:
                                if buf:
                                    chunks.append((buf, 0.08))
                                buf = w
            if buf:
                chunks.append((buf, end_pause))

    return chunks


def export_audio_mp3(
    audio: np.ndarray,
    sr: int,
    output_mp3: str | Path,
    bitrate: str = "192k",
) -> Path:
    """Encode mono float32 audio array into an MP3 file via FFmpeg."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found in PATH; cannot export MP3.")

    out_path = Path(output_mp3)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        temp_wav = Path(tf.name)

    try:
        _write_wav_pcm16(temp_wav, audio, sr)
        cmd = [
            ffmpeg, "-y", "-v", "error",
            "-i", str(temp_wav),
            "-vn",
            "-ar", "44100",
            "-ac", "1",
            "-c:a", "libmp3lame",
            "-b:a", bitrate,
            str(out_path),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode != 0 or not out_path.exists() or out_path.stat().st_size == 0:
            raise RuntimeError(f"FFmpeg MP3 export failed: {res.stderr.strip()[:300]}")
        return out_path
    finally:
        if temp_wav.exists():
            try:
                temp_wav.unlink()
            except Exception:
                pass


def synthesize_text_to_mp3(
    text: str,
    output_mp3: str | Path,
    backend: VoiceBackend,
    ref_audio_path: Optional[str | Path] = None,
    speed: float = 1.0,
    speaker_id: str = "spk_0",
    bitrate: str = "192k",
    on_progress: Optional[Callable[[float], None]] = None,
) -> dict[str, Any]:
    """Synthesize arbitrarily long text using a cloned voice backend and export MP3."""
    chunks = split_tts_chunks(text)
    if not chunks:
        raise ValueError("Nội dung văn bản đọc TTS không được để trống.")

    assembled_parts: list[np.ndarray] = []
    out_sr = 24000

    for idx, (chunk_text, pause_s) in enumerate(chunks):
        audio, sr = backend.synthesize(
            text=chunk_text,
            ref_audio_path=ref_audio_path,
            speed=1.0,
            target_duration=None,
            speaker_id=speaker_id,
        )
        if sr != out_sr and idx > 0 and len(audio) > 0:
            audio = resample_audio(audio, sr, out_sr)
        elif idx == 0:
            out_sr = sr

        if len(audio) > 0:
            fade = min(int(0.015 * out_sr), len(audio) // 4)
            if fade > 0:
                audio = audio.copy()
                audio[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
                audio[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
            assembled_parts.append(audio.astype(np.float32))

        if idx < len(chunks) - 1 and pause_s > 0:
            assembled_parts.append(np.zeros(int(pause_s * out_sr), dtype=np.float32))

        if on_progress:
            on_progress((idx + 1) / len(chunks))

    combined = np.concatenate(assembled_parts) if assembled_parts else np.zeros(int(0.5 * out_sr), dtype=np.float32)

    # Apply requested reading speed cleanly with FFmpeg atempo
    if abs(speed - 1.0) > 0.03 and len(combined) > 0:
        combined = change_audio_speed_ffmpeg(combined, out_sr, speed)

    # Gentle peak normalization to -1 dBFS
    peak = float(np.max(np.abs(combined))) if len(combined) > 0 else 0.0
    if peak > 0.01:
        combined = (combined / peak) * 0.89

    mp3_path = export_audio_mp3(combined, out_sr, output_mp3, bitrate=bitrate)
    duration_s = round(len(combined) / max(1, out_sr), 2)
    return {
        "mp3_path": str(mp3_path),
        "duration_s": duration_s,
        "chunks": len(chunks),
        "sample_rate": out_sr,
    }

