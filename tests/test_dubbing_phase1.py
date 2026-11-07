"""Unit tests for Phase 1 of Dubbing module."""

import shutil
import subprocess
import pytest
from pathlib import Path

from ecoup.dubbing.audio import (
    AudioProcessingError,
    NoAudioError,
    extract_original_hq,
    probe_audio_info,
    separate_stems,
)
from ecoup.dubbing.config import AudioExtractConfig, DemucsConfig, DubbingConfig
from ecoup.dubbing.manifest import ProjectManifest
from ecoup.dubbing.pipeline import DubbingPipeline


@pytest.fixture
def dummy_video(tmp_path: Path) -> Path:
    """Generate a tiny 1-second test MP4 with synthetic audio using ffmpeg."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg not available")

    out_file = tmp_path / "test_sample.mp4"
    cmd = [
        ffmpeg, "-y", "-v", "error",
        "-f", "lavfi", "-i", "testsrc=duration=1:size=160x120:rate=10",
        "-f", "lavfi", "-i", "sine=frequency=1000:duration=1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "64k",
        str(out_file),
    ]
    subprocess.run(cmd, check=True)
    return out_file


@pytest.fixture
def silent_video(tmp_path: Path) -> Path:
    """Generate a 1-second video without audio stream."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg not available")

    out_file = tmp_path / "silent.mp4"
    cmd = [
        ffmpeg, "-y", "-v", "error",
        "-f", "lavfi", "-i", "testsrc=duration=1:size=160x120:rate=10",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(out_file),
    ]
    subprocess.run(cmd, check=True)
    return out_file


def test_config_validation():
    cfg = AudioExtractConfig(sample_rate=44100, channels=2)
    cfg.validate()

    with pytest.raises(ValueError, match="Invalid sample_rate"):
        AudioExtractConfig(sample_rate=12345).validate()

    with pytest.raises(ValueError, match="Invalid channels"):
        AudioExtractConfig(channels=5).validate()

    demucs_cfg = DemucsConfig()
    demucs_cfg.validate()

    dub_cfg = DubbingConfig()
    dub_cfg.validate()
    h_s0 = dub_cfg.get_step_hash("S0")
    h_s1 = dub_cfg.get_step_hash("S1")
    assert len(h_s0) == 32
    assert len(h_s1) == 32
    assert h_s0 != h_s1


def test_manifest_lifecycle(tmp_path: Path):
    manifest = ProjectManifest.load_or_create(tmp_path, "dummy.mp4")
    assert manifest.manifest_path.exists()

    dummy_file = tmp_path / "out.wav"
    dummy_file.write_bytes(b"DATA")

    config_hash = "abc123hash"
    manifest.record_step("S0", config_hash, outputs={"out": "out.wav"})
    assert manifest.is_step_valid("S0", config_hash, [dummy_file]) is True

    # Invalid when config hash differs
    assert manifest.is_step_valid("S0", "different_hash", [dummy_file]) is False

    # Invalid when file is deleted
    dummy_file.unlink()
    assert manifest.is_step_valid("S0", config_hash, [dummy_file]) is False


def test_manifest_invalidate_from(tmp_path: Path):
    manifest = ProjectManifest.load_or_create(tmp_path, "dummy.mp4")
    manifest.record_step("S0", "hash0", {})
    manifest.record_step("S1", "hash1", {})
    manifest.record_step("S2", "hash2", {})

    manifest.invalidate_from("S1")
    assert "S0" in manifest.steps
    assert "S1" not in manifest.steps
    assert "S2" not in manifest.steps


def test_probe_and_extract_silent_video(silent_video: Path, tmp_path: Path):
    with pytest.raises(NoAudioError):
        probe_audio_info(silent_video)

    with pytest.raises(NoAudioError):
        extract_original_hq(silent_video, tmp_path / "out.wav")


def test_pipeline_s0_s1_mock(dummy_video: Path, tmp_path: Path):
    project_dir = tmp_path / "test_project"
    cfg = DubbingConfig(
        project_dir=str(project_dir),
        demucs=DemucsConfig(mock=True),
    )

    progress_steps = []

    def on_progress(step: str, frac: float):
        progress_steps.append((step, frac))

    pipeline = DubbingPipeline(dummy_video, config=cfg, on_progress=on_progress)

    # 1. Run S0
    orig_wav = pipeline.run_s0_intake()
    assert orig_wav.exists()
    assert orig_wav.name == "original_hq.wav"
    assert ("S0", 1.0) in progress_steps

    # 2. Run S1
    dialogue_wav, bgm_wav = pipeline.run_s1_separation()
    assert dialogue_wav.exists() and dialogue_wav.name == "dialogue.wav"
    assert bgm_wav.exists() and bgm_wav.name == "bgm.wav"
    assert ("S1", 1.0) in progress_steps

    # 3. Test caching: re-running S0 & S1 should skip without recalculation
    pipeline2 = DubbingPipeline(dummy_video, config=cfg)
    cached_s0 = pipeline2.run_s0_intake()
    assert cached_s0 == orig_wav
    cached_s1_diag, cached_s1_bgm = pipeline2.run_s1_separation()
    assert cached_s1_diag == dialogue_wav
    assert cached_s1_bgm == bgm_wav
