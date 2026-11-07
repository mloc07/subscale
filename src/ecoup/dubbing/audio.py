"""Audio extraction and stem separation for AI Dubbing module."""

from __future__ import annotations

import gc
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

from .config import AudioExtractConfig, DemucsConfig


class NoAudioError(RuntimeError):
    """Raised when the input video has no audio stream."""


class AudioProcessingError(RuntimeError):
    """Raised when audio extraction or separation fails."""


def probe_audio_info(video: str | Path) -> dict:
    """Probe audio stream information via ffprobe. Raises NoAudioError if silent."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe not found in PATH.")

    video = Path(video)
    if not video.exists():
        raise FileNotFoundError(f"Video file not found: {video}")

    cmd = [
        ffprobe, "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=codec_name,channels,sample_rate,duration:format=duration",
        "-of", "json",
        str(video),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        raise AudioProcessingError(f"ffprobe failed: {res.stderr.strip()[:300]}")

    try:
        data = json.loads(res.stdout)
    except Exception as e:
        raise AudioProcessingError(f"Failed to parse ffprobe output: {e}")

    streams = data.get("streams", [])
    if not streams:
        raise NoAudioError(f"Video has no audio stream: {video}")

    s = streams[0]
    duration_str = s.get("duration") or data.get("format", {}).get("duration", "0")
    try:
        duration = float(duration_str)
    except (TypeError, ValueError):
        duration = 0.0

    return {
        "codec_name": s.get("codec_name", "unknown"),
        "channels": int(s.get("channels", 2) or 2),
        "sample_rate": int(s.get("sample_rate", 44100) or 44100),
        "duration": duration,
    }


def extract_original_hq(
    video: str | Path,
    output_wav: str | Path,
    cfg: Optional[AudioExtractConfig] = None,
) -> Path:
    """Extract high quality PCM WAV (44.1 kHz stereo by default) from video."""
    if cfg is None:
        cfg = AudioExtractConfig()
    cfg.validate()

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found in PATH.")

    # Probe first to ensure audio exists
    probe_audio_info(video)

    output_wav = Path(output_wav)
    output_wav.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg, "-y", "-v", "error",
        "-i", str(video),
        "-vn",
        "-ac", str(cfg.channels),
        "-ar", str(cfg.sample_rate),
        "-c:a", "pcm_s16le",
        str(output_wav),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0 or not output_wav.exists() or output_wav.stat().st_size == 0:
        raise AudioProcessingError(f"FFmpeg audio extraction failed: {res.stderr.strip()[:400]}")

    return output_wav


def separate_stems(
    audio_path: str | Path,
    output_dir: str | Path,
    cfg: Optional[DemucsConfig] = None,
) -> Tuple[Path, Path]:
    """Separate dialogue and BGM (background music + SFX) using Demucs.
    
    Returns (dialogue_wav, bgm_wav).
    Raises AudioProcessingError strictly if separation fails.
    """
    if cfg is None:
        cfg = DemucsConfig()
    cfg.validate()

    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dialogue_target = output_dir / "dialogue.wav"
    bgm_target = output_dir / "bgm.wav"

    if cfg.mock:
        # Create mock WAV files for CI/unit tests without GPU/Demucs
        import wave
        for target in (dialogue_target, bgm_target):
            with wave.open(str(target), "wb") as wf:
                wf.setnchannels(2)
                wf.setsampwidth(2)
                wf.setframerate(44100)
                # 0.5s of silence
                wf.writeframes(b"\x00\x00" * 44100)
        return dialogue_target, bgm_target

    if not audio_path.exists():
        raise FileNotFoundError(f"Input audio file not found: {audio_path}")

    # Determine device
    device = cfg.device
    if device == "auto":
        device = "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
        except Exception:
            pass

    temp_demucs_dir = output_dir / ".demucs_raw"
    temp_demucs_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "demucs.separate",
        "-n", cfg.model_name,
        f"--two-stems={cfg.two_stems}",
        "-d", device,
        "-o", str(temp_demucs_dir),
        str(audio_path),
    ]

    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        raise AudioProcessingError(f"Demucs stem separation failed (code {res.returncode}): {res.stderr.strip()[:500]}")

    # Demucs outputs under temp_demucs_dir / model_name / audio_stem / (vocals.wav & no_vocals.wav)
    stem_dir = temp_demucs_dir / cfg.model_name / audio_path.stem
    raw_vocals = stem_dir / "vocals.wav"
    raw_bgm = stem_dir / "no_vocals.wav"

    if not raw_vocals.exists() or not raw_bgm.exists():
        # Search recursively if directory structure varies
        found_vocals = list(temp_demucs_dir.glob("**/vocals.wav"))
        found_bgm = list(temp_demucs_dir.glob("**/no_vocals.wav"))
        if not found_vocals or not found_bgm:
            raise AudioProcessingError(f"Demucs completed but output files missing in {temp_demucs_dir}")
        raw_vocals = found_vocals[0]
        raw_bgm = found_bgm[0]

    # Move / copy to final destination
    shutil.copy2(raw_vocals, dialogue_target)
    shutil.copy2(raw_bgm, bgm_target)

    # Free memory
    try:
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return dialogue_target, bgm_target
