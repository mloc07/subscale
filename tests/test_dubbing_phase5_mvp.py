"""Unit tests for Phase 5 MVP (Full End-to-End Pipeline S0 -> S9)."""

import json
import shutil
import subprocess
import pytest
from pathlib import Path
import numpy as np

from ecoup.dubbing.config import (
    AsrDubbingConfig,
    DemucsConfig,
    DiarizationDubbingConfig,
    DubbingConfig,
    MixingDubbingConfig,
    TranslationDubbingConfig,
    TtsDubbingConfig,
)
from ecoup.dubbing.mixing import mix_audio_tracks, read_stereo_wav, write_stereo_wav
from ecoup.dubbing.mux import export_srt_file, mux_video_with_audio
from ecoup.dubbing.pipeline import DubbingPipeline
from ecoup.dubbing.snapping import align_audio_to_target_duration
from ecoup.dubbing.translate import TranslatedDubSegment


@pytest.fixture
def dummy_video(tmp_path: Path) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg not available")

    out_file = tmp_path / "test_mvp.mp4"
    cmd = [
        ffmpeg, "-y", "-v", "error",
        "-f", "lavfi", "-i", "testsrc=duration=3:size=160x120:rate=10",
        "-f", "lavfi", "-i", "sine=frequency=1000:duration=3",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "64k",
        str(out_file),
    ]
    subprocess.run(cmd, check=True)
    return out_file


def test_snapping_duration_alignment():
    sr = 16000
    # 2.0s audio target 3.0s -> padding
    audio_short = np.ones(2 * sr, dtype=np.float32) * 0.1
    padded, info = align_audio_to_target_duration(audio_short, sr, target_duration=3.0)
    assert len(padded) == 3 * sr
    assert info["padded_sec"] > 0.9

    # 4.0s audio target 3.2s -> speedup
    audio_long = np.ones(4 * sr, dtype=np.float32) * 0.1
    aligned, info_speed = align_audio_to_target_duration(audio_long, sr, target_duration=3.2, max_speedup=1.3)
    assert len(aligned) == int(3.2 * sr)
    assert info_speed["applied_speed"] > 1.0


def test_audio_mixing_ducking(tmp_path: Path):
    sr = 44100
    bgm_path = tmp_path / "bgm.wav"
    out_mix = tmp_path / "mixed.wav"

    # Create 4s BGM stereo
    t = np.linspace(0, 4, 4 * sr, endpoint=False)
    bgm_data = np.column_stack([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 440 * t)]).astype(np.float32) * 0.5
    write_stereo_wav(bgm_path, bgm_data, sr)

    # 1s vocal clip at t=1.0s
    vocal = np.ones(1 * sr, dtype=np.float32) * 0.3
    vocal_clips = [{"start": 1.0, "end": 2.0, "audio": vocal, "sr": sr}]

    mixed_file = mix_audio_tracks(bgm_path, vocal_clips, out_mix, ducking_db=-6.0, target_sr=sr)
    assert mixed_file.exists()
    mixed_data, m_sr = read_stereo_wav(mixed_file)
    assert m_sr == sr
    assert len(mixed_data) == len(bgm_data)


def test_export_srt(tmp_path: Path):
    srt_file = tmp_path / "test.srt"
    segs = [
        TranslatedDubSegment(id=0, start=1.0, end=3.5, duration=2.5, speaker_id="spk_0", original_text="Hello", translated_text="Xin chào", syllable_count=2, max_budget=8),
        TranslatedDubSegment(id=1, start=4.0, end=6.0, duration=2.0, speaker_id="spk_0", original_text="Goodbye", translated_text="Tạm biệt", syllable_count=2, max_budget=7),
    ]
    export_srt_file(segs, srt_file)
    assert srt_file.exists()
    content = srt_file.read_text(encoding="utf-8")
    assert "00:00:01,000 --> 00:00:03,500" in content
    assert "Xin chào" in content
    assert "Tạm biệt" in content


def test_mux_video(dummy_video: Path, tmp_path: Path):
    sr = 44100
    dummy_wav = tmp_path / "dub_mix.wav"
    t = np.linspace(0, 3, 3 * sr, endpoint=False)
    audio = np.column_stack([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 440 * t)]).astype(np.float32) * 0.2
    write_stereo_wav(dummy_wav, audio, sr)

    out_mp4 = tmp_path / "final.mp4"
    mux_video_with_audio(dummy_video, dummy_wav, out_mp4)
    assert out_mp4.exists()
    assert out_mp4.stat().st_size > 1000


def test_full_pipeline_mvp_mock(dummy_video: Path, tmp_path: Path):
    project_dir = tmp_path / "mvp_project"
    cfg = DubbingConfig(
        project_dir=str(project_dir),
        demucs=DemucsConfig(mock=True),
        asr=AsrDubbingConfig(mock=True),
        diarization=DiarizationDubbingConfig(mock=True, num_speakers=1),
        translation=TranslationDubbingConfig(mock=True),
        tts=TtsDubbingConfig(mock=True),
        mixing=MixingDubbingConfig(target_sr=44100),
    )

    pipeline = DubbingPipeline(dummy_video, config=cfg)
    result = pipeline.run_full_pipeline(context_notes="Test MVP")

    assert result["status"] == "completed"
    assert Path(result["video_path"]).exists()
    assert Path(result["srt_path"]).exists()

    # Check all key files exist
    assert (project_dir / "manifest.json").exists()
    assert (project_dir / "audio" / "original_hq.wav").exists()
    assert (project_dir / "audio" / "dialogue.wav").exists()
    assert (project_dir / "audio" / "bgm.wav").exists()
    assert (project_dir / "transcript" / "segments.json").exists()
    assert (project_dir / "transcript" / "diarized.json").exists()
    assert (project_dir / "voices" / "spk_0" / "ref_clean.wav").exists()
    assert (project_dir / "translation" / "translated.json").exists()
    assert (project_dir / "mix" / "dub_mix.wav").exists()
