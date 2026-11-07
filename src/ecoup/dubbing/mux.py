"""Video muxing without re-encoding (S9)."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Sequence

from .translate import TranslatedDubSegment


def export_srt_file(segments: Sequence[TranslatedDubSegment], output_srt: str | Path) -> Path:
    """Generate a standard SubRip (.srt) subtitle file in Vietnamese."""
    output_srt = Path(output_srt)
    output_srt.parent.mkdir(parents=True, exist_ok=True)

    def format_ts(sec: float) -> str:
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = int(sec % 60)
        ms = int(round((sec - int(sec)) * 1000))
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    lines = []
    for idx, s in enumerate(segments, start=1):
        lines.append(str(idx))
        lines.append(f"{format_ts(s.start)} --> {format_ts(s.end)}")
        lines.append(s.translated_text)
        lines.append("")

    with open(output_srt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return output_srt


def mux_video_with_audio(
    video_path: str | Path,
    audio_path: str | Path,
    output_video_path: str | Path,
) -> Path:
    """Replace video audio track with dubbed audio using FFmpeg stream copy (-c:v copy)."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found in PATH.")

    video_path = Path(video_path)
    audio_path = Path(audio_path)
    output_video_path = Path(output_video_path)
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg, "-y", "-v", "error",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        str(output_video_path),
    ]

    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0 or not output_video_path.exists() or output_video_path.stat().st_size == 0:
        raise RuntimeError(f"FFmpeg muxing failed: {res.stderr.strip()[:400]}")

    return output_video_path
