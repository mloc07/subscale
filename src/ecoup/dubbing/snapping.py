"""Time snapping and duration adjustment (S7)."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Optional

import numpy as np

from .diarization import _write_wav_pcm16


def change_audio_speed_ffmpeg(audio: np.ndarray, sr: int, speed: float) -> np.ndarray:
    """Change audio speed using FFmpeg atempo filter (preserves pitch)."""
    if abs(speed - 1.0) < 0.02 or len(audio) == 0:
        return audio

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        # Fallback to linear resample
        num_samples = int(len(audio) / speed)
        old_indices = np.linspace(0, len(audio) - 1, num=len(audio))
        new_indices = np.linspace(0, len(audio) - 1, num=num_samples)
        return np.interp(new_indices, old_indices, audio).astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf_in:
        in_path = tf_in.name
    out_path = in_path.replace(".wav", "_out.wav")

    try:
        _write_wav_pcm16(Path(in_path), audio, sr)
        # atempo supports 0.5 to 2.0
        safe_speed = max(0.5, min(2.0, speed))
        cmd = [
            ffmpeg, "-y", "-v", "error",
            "-i", in_path,
            "-filter:a", f"atempo={safe_speed:.3f}",
            "-vn", "-c:a", "pcm_s16le",
            out_path,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode == 0 and os.path.exists(out_path):
            with wave.open(out_path, "rb") as wf:
                raw = wf.readframes(wf.getnframes())
                return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        return audio
    finally:
        for p in (in_path, out_path):
            if os.path.exists(p):
                try:
                    os.unlink(p)
                except Exception:
                    pass


def align_audio_to_target_duration(
    audio: np.ndarray,
    sr: int,
    target_duration: float,
    min_speed: float = 1.0,
    max_speedup: float = 1.35,
) -> tuple[np.ndarray, dict[str, float]]:
    """Align audio to target duration by dynamic tempo adjustment and balanced padding."""
    actual_dur = len(audio) / sr
    target_samples = int(target_duration * sr)
    info = {
        "original_dur": round(actual_dur, 3),
        "target_dur": round(target_duration, 3),
        "applied_speed": 1.0,
        "padded_sec": 0.0,
    }

    if target_samples <= 0 or len(audio) == 0:
        return audio, info

    # Bidirectional tempo adjustment if discrepancy is greater than 80ms
    if (min_speed < 1.0 and actual_dur < target_duration - 0.08) or (actual_dur > target_duration + 0.08):
        raw_speed = actual_dur / target_duration
        effective_speed = max(min_speed, min(max_speedup, raw_speed))
        if abs(effective_speed - 1.0) > 0.02:
            audio = change_audio_speed_ffmpeg(audio, sr, effective_speed)
            info["applied_speed"] = round(effective_speed, 3)

    # Pad with silence if shorter than target (balanced padding: small front, rest back)
    current_samples = len(audio)
    if current_samples < target_samples:
        pad_len = target_samples - current_samples
        pad_pre = min(int(0.08 * sr), pad_len // 4)
        pad_post = pad_len - pad_pre
        padded = np.pad(audio, (pad_pre, pad_post), mode="constant")
        info["padded_sec"] = round(pad_len / sr, 3)
        return padded, info

    # If slightly longer due to max_speedup limit, trim with 20ms fade-out
    if len(audio) > target_samples:
        trimmed = audio[:target_samples].copy()
        fade_len = min(int(0.02 * sr), target_samples)
        if fade_len > 0:
            fade = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
            trimmed[-fade_len:] *= fade
        return trimmed, info

    return audio, info
