"""Audio extraction via FFmpeg."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class NoAudioError(RuntimeError):
    """Raised when the input video has no audio stream."""


def has_audio(video: str | Path) -> bool:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe not found in PATH.")
    res = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True, check=False,
    )
    return bool(res.stdout.strip())


def video_info(video: str | Path) -> dict:
    """Return width/height/duration of the first video stream."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe not found in PATH.")
    res = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height:format=duration",
         "-of", "default=nw=1", str(video)],
        capture_output=True, text=True, check=False,
    )
    info: dict = {}
    for line in res.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            info[k.strip()] = v.strip()
    return {
        "width": int(info.get("width", 0) or 0),
        "height": int(info.get("height", 0) or 0),
        "duration": float(info.get("duration", 0) or 0),
    }


def extract_audio(video: str | Path, wav_path: str | Path, sample_rate: int = 16000) -> Path:
    """Extract mono 16 kHz PCM WAV. Raises ``NoAudioError`` for silent videos (never hangs)."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found in PATH.")
    if not Path(video).exists():
        raise FileNotFoundError(f"Video not found: {video}")
    if not has_audio(video):
        raise NoAudioError(f"Video has no audio stream: {video}")
    wav_path = Path(wav_path)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    res = subprocess.run(
        [ffmpeg, "-y", "-v", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", str(sample_rate),
         "-c:a", "pcm_s16le", str(wav_path)],
        capture_output=True, text=True, check=False, timeout=3600,
    )
    if res.returncode != 0 or not wav_path.exists():
        raise RuntimeError(f"FFmpeg audio extraction failed: {res.stderr.strip()[:500]}")
    return wav_path


def extract_vocals(media_path: str | Path, wav_path: str | Path, sample_rate: int = 16000) -> Path:
    """Isolate vocals using Demucs AI and resample to mono 16 kHz PCM WAV.
    
    Tối ưu cho bài hát, MV có beat dày, nhạc nền lớn đè giọng hát ca sĩ.
    """
    import sys
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found in PATH.")

    media_path = Path(media_path)
    if not media_path.exists():
        raise FileNotFoundError(f"Media file not found: {media_path}")

    wav_path = Path(wav_path)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    temp_demucs_dir = wav_path.parent / "demucs_cache"
    temp_demucs_dir.mkdir(parents=True, exist_ok=True)

    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
    except Exception:
        pass

    cmd = [
        sys.executable, "-m", "demucs.separate",
        "-n", "htdemucs", "--two-stems=vocals",
        "-d", device,
        "-o", str(temp_demucs_dir),
        str(media_path)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"Demucs vocal separation failed: {res.stderr.strip()[:400]}")

    # Output is typically in temp_demucs_dir / "htdemucs" / stem / "vocals.wav"
    raw_vocal = temp_demucs_dir / "htdemucs" / media_path.stem / "vocals.wav"
    if not raw_vocal.exists():
        candidates = list(temp_demucs_dir.glob("**/vocals.wav"))
        if not candidates:
            raise RuntimeError(f"Demucs did not output vocals.wav in {temp_demucs_dir}")
        raw_vocal = candidates[0]

    # Convert to 16kHz mono pcm_s16le
    res_ff = subprocess.run(
        [ffmpeg, "-y", "-v", "error", "-i", str(raw_vocal), "-vn", "-ac", "1", "-ar", str(sample_rate),
         "-c:a", "pcm_s16le", str(wav_path)],
        capture_output=True, text=True, check=False,
    )
    if res_ff.returncode != 0 or not wav_path.exists():
        raise RuntimeError(f"FFmpeg conversion of vocals failed: {res_ff.stderr.strip()[:400]}")

    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return wav_path
