"""Audio mixing, Ducking, and Loudness Normalization (S8)."""

from __future__ import annotations

import os
import wave
from pathlib import Path
from typing import Sequence

import numpy as np

from .asr import resample_audio


def read_stereo_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        n_ch = wf.getnchannels()
        sw = wf.getsampwidth()
        sr = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    if sw == 2:
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        data = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sw}")

    if n_ch == 1:
        stereo = np.column_stack([data, data])
    else:
        stereo = data.reshape(-1, n_ch)[:, :2]

    return stereo, sr


def write_stereo_wav(path: Path, audio: np.ndarray, sr: int = 44100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio, -0.98, 0.98)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def mix_audio_tracks(
    bgm_path: str | Path,
    vocal_clips: list[dict],  # list of {"start": float, "end": float, "audio": np.ndarray, "sr": int}
    output_path: str | Path,
    ducking_db: float = -8.0,
    target_sr: int = 44100,
) -> Path:
    """Mix vocal clips into BGM with ducking and volume normalization."""
    bgm_path = Path(bgm_path)
    output_path = Path(output_path)

    bgm, bgm_sr = read_stereo_wav(bgm_path)
    if bgm_sr != target_sr:
        # Resample both channels
        ch0 = resample_audio(bgm[:, 0], bgm_sr, target_sr)
        ch1 = resample_audio(bgm[:, 1], bgm_sr, target_sr)
        bgm = np.column_stack([ch0, ch1])

    total_samples = len(bgm)
    dialogue_track = np.zeros_like(bgm, dtype=np.float32)

    # Ducking gain envelope: default 1.0 (0dB)
    ducking_gain = np.ones(total_samples, dtype=np.float32)
    duck_factor = 10.0 ** (ducking_db / 20.0)  # e.g. -8dB -> ~0.4

    # Smooth transition length (50ms)
    ramp_len = int(0.05 * target_sr)
    ramp_down = np.linspace(1.0, duck_factor, ramp_len, dtype=np.float32)
    ramp_up = np.linspace(duck_factor, 1.0, ramp_len, dtype=np.float32)

    for clip in vocal_clips:
        start_sec = clip["start"]
        audio = clip["audio"]
        clip_sr = clip.get("sr", target_sr)

        if clip_sr != target_sr:
            audio = resample_audio(audio, clip_sr, target_sr)

        # Convert mono vocal to stereo
        if audio.ndim == 1:
            v_stereo = np.column_stack([audio, audio])
        else:
            v_stereo = audio

        st_idx = int(start_sec * target_sr)
        en_idx = st_idx + len(v_stereo)

        if st_idx >= total_samples:
            continue

        valid_len = min(len(v_stereo), total_samples - st_idx)
        dialogue_track[st_idx : st_idx + valid_len] += v_stereo[:valid_len]

        # Apply ducking around dialogue
        d_st = max(0, st_idx - ramp_len)
        d_en = min(total_samples, en_idx + ramp_len)

        # Duck the interior
        ducking_gain[st_idx:en_idx] = np.minimum(ducking_gain[st_idx:en_idx], duck_factor)

    # Apply ducking to BGM
    bgm_ducked = bgm * ducking_gain[:, None]

    # Combine tracks
    mixed = bgm_ducked + dialogue_track

    # Normalize peak if clipping
    peak = float(np.max(np.abs(mixed)))
    if peak > 0.95:
        mixed = mixed * (0.95 / peak)

    write_stereo_wav(output_path, mixed, target_sr)
    return output_path
