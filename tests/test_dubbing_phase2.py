"""Unit tests for Phase 2 of Dubbing module (ASR & Diarization)."""

import json
import shutil
import subprocess
import pytest
from pathlib import Path

from ecoup.dubbing.asr import AsrOptions, DubbingSegment, is_hallucination, transcribe_dialogue
from ecoup.dubbing.config import (
    AsrDubbingConfig,
    DemucsConfig,
    DiarizationDubbingConfig,
    DubbingConfig,
)
from ecoup.dubbing.diarization import DiarizationOptions, SpeakerDiarizer, export_speaker_previews
from ecoup.dubbing.pipeline import DubbingPipeline


@pytest.fixture
def dummy_video(tmp_path: Path) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg not available")

    out_file = tmp_path / "test_sample.mp4"
    cmd = [
        ffmpeg, "-y", "-v", "error",
        "-f", "lavfi", "-i", "testsrc=duration=2:size=160x120:rate=10",
        "-f", "lavfi", "-i", "sine=frequency=1000:duration=2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "64k",
        str(out_file),
    ]
    subprocess.run(cmd, check=True)
    return out_file


def test_hallucination_filtering():
    assert is_hallucination("Thanks for watching!") is True
    assert is_hallucination("Thank you for watching this video.") is True
    assert is_hallucination("Like and subscribe!") is True
    assert is_hallucination("Hãy subscribe cho kênh của mình nhé") is True
    assert is_hallucination("How do you know what is inside you?") is False


def test_transcribe_dialogue_mock(tmp_path: Path):
    dummy_wav = tmp_path / "dialogue.wav"
    dummy_wav.write_bytes(b"DATA")

    opts = AsrOptions(mock=True)
    segments, lang, prob = transcribe_dialogue(dummy_wav, options=opts)
    assert len(segments) == 2
    assert lang == "en"
    assert prob > 0.9
    assert segments[0].text == "Hello world, testing dubbing."
    assert segments[0].duration == 2.5


def test_diarization_clustering_mock(tmp_path: Path):
    # Create synthetic mono 16kHz WAV
    dummy_wav = tmp_path / "dialogue.wav"
    import wave
    with wave.open(str(dummy_wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 16000 * 5)  # 5 seconds

    segments = [
        DubbingSegment(id=0, start=0.5, end=2.0, text="First person"),
        DubbingSegment(id=1, start=1.9, end=3.5, text="Second overlapping person"),
        DubbingSegment(id=2, start=3.8, end=4.8, text="Third person"),
    ]

    diarizer = SpeakerDiarizer(DiarizationOptions(mock=True, num_speakers=2))
    diarized = diarizer.diarize_segments(dummy_wav, segments)

    assert len(diarized) == 3
    for s in diarized:
        assert s.speaker_id is not None
        assert s.speaker_id.startswith("spk_")
    # Check overlap detection
    assert diarized[0].is_overlap is True
    assert diarized[1].is_overlap is True
    assert diarized[2].is_overlap is False


def test_pipeline_s2_s3_mock(dummy_video: Path, tmp_path: Path):
    project_dir = tmp_path / "dub_project"
    cfg = DubbingConfig(
        project_dir=str(project_dir),
        demucs=DemucsConfig(mock=True),
        asr=AsrDubbingConfig(mock=True),
        diarization=DiarizationDubbingConfig(mock=True, num_speakers=1),
    )

    pipeline = DubbingPipeline(dummy_video, config=cfg)

    # 1. Run S2
    segs = pipeline.run_s2_asr()
    assert len(segs) == 2
    seg_file = project_dir / "transcript" / "segments.json"
    assert seg_file.exists()

    # 2. Run S3
    diar_segs = pipeline.run_s3_diarization()
    assert len(diar_segs) == 2
    diar_file = project_dir / "transcript" / "diarized.json"
    assert diar_file.exists()

    # Check preview export
    spk_0_preview = project_dir / "voices" / "spk_0" / "preview.wav"
    spk_0_profile = project_dir / "voices" / "spk_0" / "profile.json"
    assert spk_0_preview.exists()
    assert spk_0_profile.exists()

    # 3. Test caching: second run of S2 and S3 must be instant
    pipeline2 = DubbingPipeline(dummy_video, config=cfg)
    cached_s2 = pipeline2.run_s2_asr()
    cached_s3 = pipeline2.run_s3_diarization()
    assert len(cached_s2) == 2
    assert len(cached_s3) == 2
