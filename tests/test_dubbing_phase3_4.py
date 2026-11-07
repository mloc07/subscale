"""Unit tests for Phase 3 (Voice Profile) and Phase 4 (Translation)."""

import json
import shutil
import subprocess
import pytest
from pathlib import Path
import numpy as np

from ecoup.dubbing.asr import DubbingSegment
from ecoup.dubbing.config import (
    AsrDubbingConfig,
    DemucsConfig,
    DiarizationDubbingConfig,
    DubbingConfig,
    TranslationDubbingConfig,
)
from ecoup.dubbing.pipeline import DubbingPipeline
from ecoup.dubbing.translate import (
    DubbingTranslator,
    TranslationConfig,
    count_vietnamese_syllables,
)
from ecoup.dubbing.voice_profile import (
    VoiceProfile,
    build_voice_profiles,
    calculate_audio_quality,
)


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


def test_audio_quality_calculation():
    sr = 16000
    # Short audio (0.5s)
    short_audio = np.random.randn(int(0.5 * sr)) * 0.1
    score, flags = calculate_audio_quality(short_audio, sr)
    assert "sample_too_short" in flags
    assert score <= 0.6

    # Normal 3s audio
    normal_audio = np.random.randn(int(3.0 * sr)) * 0.1
    score, flags = calculate_audio_quality(normal_audio, sr)
    assert score >= 0.7


def test_syllable_counter():
    assert count_vietnamese_syllables("Xin chào Việt Nam!") == 4
    assert count_vietnamese_syllables("Đừng chống đẩy một cái, hãy trăm cái luôn.") == 9
    assert count_vietnamese_syllables("") == 0


def test_dubbing_translator_mock():
    cfg = TranslationConfig(mock=True, syllables_per_second=3.8)
    translator = DubbingTranslator(cfg)

    segs = [
        DubbingSegment(id=0, start=0.0, end=3.0, text="Don't do one push up, do one hundred.", speaker_id="spk_0"),
        DubbingSegment(id=1, start=3.5, end=7.5, text="Run a marathon.", speaker_id="spk_0"),
    ]

    translated = translator.translate_segments(segs)
    assert len(translated) == 2
    assert translated[0].id == 0
    assert translated[0].max_budget == int(3.0 * 3.8)
    assert translated[0].speaker_id == "spk_0"
    assert translated[0].syllable_count > 0


def test_pipeline_s4_s5_mock(dummy_video: Path, tmp_path: Path):
    project_dir = tmp_path / "dub_project"
    cfg = DubbingConfig(
        project_dir=str(project_dir),
        demucs=DemucsConfig(mock=True),
        asr=AsrDubbingConfig(mock=True),
        diarization=DiarizationDubbingConfig(mock=True, num_speakers=1),
        translation=TranslationDubbingConfig(mock=True),
    )

    pipeline = DubbingPipeline(dummy_video, config=cfg)

    # 1. Run S4
    profiles = pipeline.run_s4_voice_profiles()
    assert len(profiles) >= 1
    assert "spk_0" in profiles
    assert profiles["spk_0"].quality_score > 0.0

    ref_wav = project_dir / "voices" / "spk_0" / "ref_clean.wav"
    assert ref_wav.exists()

    # 2. Run S5
    translated = pipeline.run_s5_translation()
    assert len(translated) == 2
    trans_json = project_dir / "translation" / "translated.json"
    assert trans_json.exists()

    # 3. Test caching for S4 & S5
    pipeline2 = DubbingPipeline(dummy_video, config=cfg)
    c_prof = pipeline2.run_s4_voice_profiles()
    c_trans = pipeline2.run_s5_translation()
    assert len(c_prof) == len(profiles)
    assert len(c_trans) == len(translated)
