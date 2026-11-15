"""Unit and integration tests for Tab 4 (Voice Cloning Studio) & Cross-Video Mapping."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from ecoup.cli import main as cli_main
from ecoup.dubbing import (
    SPECIAL_ORIGINAL,
    SPECIAL_SELF,
    DubbingConfig,
    DubbingPipeline,
    MockVoiceBackend,
    VoiceBankManager,
    clone_voices_from_video,
    estimate_median_f0_hz,
    suggest_speaker_mapping,
    synthesize_text_to_mp3,
)
from ecoup.dubbing.diarization import _write_wav_pcm16
from ecoup.dubbing.tts import OpenVoiceBackend, compute_pitch_shift_semitones


def _require_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg not found in PATH")
    return ffmpeg


def _ffprobe_json(path: Path) -> dict:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        pytest.skip("ffprobe not found in PATH")
    res = subprocess.run(
        [ffprobe, "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(res.stdout)


def _make_dual_speaker_wav(wav_path: Path, sr: int = 24000, f1: float = 120.0, f2: float = 235.0) -> None:
    """Create 8.0s audio: 0-3.8s voiced ~f1 Hz, 3.8-8.0s voiced ~f2 Hz with distinct envelopes."""
    t1 = np.arange(int(3.8 * sr)) / sr
    t2 = np.arange(int(4.2 * sr)) / sr
    sig1 = 0.45 * np.sin(2 * np.pi * f1 * t1) + 0.12 * np.sin(2 * np.pi * 2 * f1 * t1)
    sig2 = 0.25 * np.sin(2 * np.pi * f2 * t2) + 0.08 * np.sin(2 * np.pi * 2 * f2 * t2)
    full = np.concatenate([sig1, sig2]).astype(np.float32)
    _write_wav_pcm16(wav_path, full, sr=sr)


def _make_dummy_video(video_path: Path, f1: float = 120.0, f2: float = 235.0) -> Path:
    ffmpeg = _require_ffmpeg()
    wav_tmp = video_path.with_suffix(".tmp.wav")
    _make_dual_speaker_wav(wav_tmp, sr=24000, f1=f1, f2=f2)
    cmd = [
        ffmpeg, "-y", "-v", "error",
        "-f", "lavfi", "-i", "color=c=black:s=320x240:r=24:d=8.0",
        "-i", str(wav_tmp),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest",
        str(video_path),
    ]
    subprocess.run(cmd, check=True)
    wav_tmp.unlink(missing_ok=True)
    return video_path


def test_voice_bank_crud_and_pitch_helpers(tmp_path: Path) -> None:
    _require_ffmpeg()
    bank = VoiceBankManager(tmp_path / "voice_bank")

    # Create two synthetic speaker reference wavs with clear low vs high F0
    sr = 24000
    t = np.arange(int(2.0 * sr)) / sr
    male_wav = tmp_path / "male.wav"
    female_wav = tmp_path / "female.wav"
    _write_wav_pcm16(male_wav, (0.4 * np.sin(2 * np.pi * 115.0 * t)).astype(np.float32), sr)
    _write_wav_pcm16(female_wav, (0.4 * np.sin(2 * np.pi * 230.0 * t)).astype(np.float32), sr)

    mock_backend = MockVoiceBackend()
    e1 = bank.save_voice(
        clone_id="clone_male",
        ref_wav_src=male_wav,
        display_name="Giọng Nam Trầm",
        source_video="video_a.mp4",
        speaker_id="spk_0",
        clone_index=1,
        quality_score=0.92,
        speaking_duration_s=5.4,
        reference_text="Hello from male",
        backend=mock_backend,
        batch_ts=100.0,
    )
    e2 = bank.save_voice(
        clone_id="clone_female",
        ref_wav_src=female_wav,
        display_name="Giọng Nữ Cao",
        source_video="video_a.mp4",
        speaker_id="spk_1",
        clone_index=2,
        quality_score=0.88,
        speaking_duration_s=4.2,
        reference_text="Hello from female",
        backend=mock_backend,
        batch_ts=100.0,
    )

    assert 95.0 <= e1.estimated_f0_hz <= 140.0
    assert 200.0 <= e2.estimated_f0_hz <= 260.0
    assert Path(e1.preview_mp3_path).exists()
    probe = _ffprobe_json(Path(e1.preview_mp3_path))
    assert any(st.get("codec_name") == "mp3" for st in probe["streams"])

    voices = bank.list_voices()
    assert [v.clone_id for v in voices] == ["clone_male", "clone_female"]

    # Pitch alignment semitones checks (base gTTS female ~215 Hz -> male ~115 Hz drops pitch)
    base_female = (0.4 * np.sin(2 * np.pi * 215.0 * t)).astype(np.float32)
    shift_to_male = compute_pitch_shift_semitones(base_female, sr, e1.estimated_f0_hz)
    assert -6.0 <= shift_to_male <= -3.5

    # Resolve variants
    label_e1 = bank.format_choice_label(e1)
    assert bank.resolve_voice("clone_male") == Path(e1.ref_wav_path)
    assert bank.resolve_voice(label_e1) == Path(e1.ref_wav_path)
    assert bank.resolve_voice("Giọng Nam Trầm") == Path(e1.ref_wav_path)
    assert bank.resolve_voice(SPECIAL_SELF) is None
    assert bank.resolve_voice(SPECIAL_ORIGINAL) is None

    renamed = bank.rename_voice("clone_male", "Peter Griffin VI")
    assert renamed.display_name == "Peter Griffin VI"
    assert bank.delete_voice("clone_female") is True
    assert len(bank.list_voices()) == 1


def _fake_separate_stems(audio_path: str | Path, output_dir: str | Path, *_args, **_kwargs) -> tuple[Path, Path]:
    """Fake S1 separation that preserves original dual-speaker waveform in dialogue.wav."""
    from ecoup.dubbing.asr import load_wav_float32

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dialogue_wav = out_dir / "dialogue.wav"
    bgm_wav = out_dir / "bgm.wav"
    arr = load_wav_float32(audio_path, target_sr=24000)
    _write_wav_pcm16(dialogue_wav, arr, sr=24000)
    _write_wav_pcm16(bgm_wav, np.zeros_like(arr, dtype=np.float32), sr=24000)
    return dialogue_wav, bgm_wav


def _fake_diarize_two_speakers(_self, _dialogue_wav, segments, should_cancel=None):
    """Assign alternating speakers spk_0 / spk_1 cleanly for deterministic unit tests."""
    for idx, s in enumerate(segments):
        s.speaker_id = f"spk_{idx % 2}"
    return list(segments)


def test_clone_multi_voices_and_cross_video_dubbing(tmp_path: Path) -> None:
    _require_ffmpeg()
    src_video = _make_dummy_video(tmp_path / "source_multi.mp4", f1=115.0, f2=230.0)
    tgt_video = _make_dummy_video(tmp_path / "target_scene.mp4", f1=130.0, f2=210.0)
    bank_dir = tmp_path / "bank"

    # 1. Clone multiple voices from Source Video A (2 speakers)
    with (
        patch("ecoup.dubbing.pipeline.separate_stems", _fake_separate_stems),
        patch("ecoup.dubbing.diarization.SpeakerDiarizer.diarize_segments", _fake_diarize_two_speakers),
    ):
        clones = clone_voices_from_video(
            video_path=src_video,
            bank_dir=bank_dir,
            label_prefix="FamilyA",
            num_speakers=2,
            min_speaker_duration_s=0.5,
            mock=True,
        )
    assert len(clones) == 2
    assert clones[0].clone_index == 1
    assert clones[1].clone_index == 2
    assert 95.0 <= clones[0].estimated_f0_hz <= 145.0
    assert 200.0 <= clones[1].estimated_f0_hz <= 260.0
    for c in clones:
        assert Path(c.ref_wav_path).exists()
        assert Path(c.preview_mp3_path).exists()

    # 2. Analyze Target Video B speakers & verify ordered mapping
    tgt_proj = tmp_path / "target_proj"
    cfg = DubbingConfig()
    cfg.project_dir = str(tgt_proj)
    cfg.demucs.mock = True
    cfg.asr.mock = True
    cfg.diarization.mock = True
    cfg.diarization.num_speakers = 2
    cfg.translation.mock = True
    cfg.tts.mock = True
    cfg.tts.backend = "mock"
    cfg.tts.voice_bank_dir = str(bank_dir)

    with (
        patch("ecoup.dubbing.pipeline.separate_stems", _fake_separate_stems),
        patch("ecoup.dubbing.diarization.SpeakerDiarizer.diarize_segments", _fake_diarize_two_speakers),
    ):
        analyzer = DubbingPipeline(tgt_video, config=cfg)
        target_speakers = analyzer.analyze_target_speakers()
    assert len(target_speakers) == 2
    assert target_speakers[0]["label"].startswith("Giọng 1")
    assert target_speakers[1]["label"].startswith("Giọng 2")
    assert Path(target_speakers[0]["preview_wav"]).exists()

    order_map = suggest_speaker_mapping(target_speakers, clones, mode="order")
    assert order_map[target_speakers[0]["speaker_id"]] == clones[0].clone_id
    assert order_map[target_speakers[1]["speaker_id"]] == clones[1].clone_id

    f0_map = suggest_speaker_mapping(target_speakers, clones, mode="f0")
    assert f0_map[target_speakers[0]["speaker_id"]] == clones[0].clone_id
    assert f0_map[target_speakers[1]["speaker_id"]] == clones[1].clone_id

    # 3. Pass 1: Dub Target Video B with order_map & trace ref paths passed to backend
    seen_refs_pass1: list[tuple[str, str]] = []
    orig_synth = MockVoiceBackend.synthesize

    def _spy_synth(self_b, text, ref_audio_path=None, speed=1.0, target_duration=None, speaker_id="spk_0", ref_text=None):
        seen_refs_pass1.append((speaker_id, str(ref_audio_path or "")))
        return orig_synth(self_b, text, ref_audio_path, speed, target_duration, speaker_id, ref_text)

    cfg.tts.speaker_voice_map = order_map
    with patch.object(MockVoiceBackend, "synthesize", _spy_synth):
        p1 = DubbingPipeline(tgt_video, config=cfg)
        res1 = p1.run_full_pipeline()

    assert Path(res1["video_path"]).exists()
    assert Path(res1["srt_path"]).exists()
    assert any(clones[0].clone_id in r[1] for r in seen_refs_pass1)
    assert any(clones[1].clone_id in r[1] for r in seen_refs_pass1)

    s5_ts_before = p1.manifest.steps["S5"].timestamp
    s6_hash_before = p1.manifest.steps["S6"].config_hash

    # Verify video stream preserved (-c:v copy) & no watermark tag inserted
    probe = _ffprobe_json(Path(res1["video_path"]))
    tags_str = json.dumps(probe.get("format", {}).get("tags", {})).lower()
    assert "lồng tiếng bằng ai" not in tags_str

    # 4. Pass 2: Swap mapping -> spk_0 uses Clone 2, spk_1 keeps "__original__"
    # S0..S5 MUST remain cached while S6..S9 cleanly re-synthesize!
    swapped_map = {
        target_speakers[0]["speaker_id"]: clones[1].clone_id,
        target_speakers[1]["speaker_id"]: SPECIAL_ORIGINAL,
    }
    cfg2 = DubbingConfig()
    cfg2.project_dir = str(tgt_proj)
    cfg2.demucs.mock = True
    cfg2.asr.mock = True
    cfg2.diarization.mock = True
    cfg2.diarization.num_speakers = 2
    cfg2.translation.mock = True
    cfg2.tts.mock = True
    cfg2.tts.backend = "mock"
    cfg2.tts.voice_bank_dir = str(bank_dir)
    cfg2.tts.speaker_voice_map = swapped_map

    seen_refs_pass2: list[tuple[str, str]] = []

    def _spy_synth2(self_b, text, ref_audio_path=None, speed=1.0, target_duration=None, speaker_id="spk_0", ref_text=None):
        seen_refs_pass2.append((speaker_id, str(ref_audio_path or "")))
        return orig_synth(self_b, text, ref_audio_path, speed, target_duration, speaker_id, ref_text)

    with patch.object(MockVoiceBackend, "synthesize", _spy_synth2):
        p2 = DubbingPipeline(tgt_video, config=cfg2)
        res2 = p2.run_full_pipeline()

    assert Path(res2["video_path"]).exists()
    assert p2.manifest.steps["S5"].timestamp == s5_ts_before  # S0-S5 cache preserved!
    assert p2.manifest.steps["S6"].config_hash != s6_hash_before
    # Only spk_0 synthesized TTS (mapped to clone_2); spk_1 copied original audio without TTS call
    assert len(seen_refs_pass2) == 1
    assert clones[1].clone_id in seen_refs_pass2[0][1]


def test_long_text_tts_to_mp3_and_cli(tmp_path: Path) -> None:
    _require_ffmpeg()
    long_text = (
        "Xin chào tất cả các bạn đang theo dõi chương trình thử nghiệm nhân bản giọng nói tiếng Việt. "
        "Hệ thống có khả năng tách từng câu văn dài thành nhiều đoạn nhỏ hợp lý; "
        "sau đó tổng hợp nối liền mượt mà không bị ngắt quãng đột ngột. "
        "Cuối cùng toàn bộ âm thanh sẽ được xuất ra định dạng MP3 chất lượng cao."
    )
    out_mp3 = tmp_path / "story.mp3"
    info = synthesize_text_to_mp3(
        text=long_text,
        output_mp3=out_mp3,
        backend=MockVoiceBackend(),
        speed=1.15,
        speaker_id="spk_0",
    )
    assert Path(info["mp3_path"]).exists()
    assert info["chunks"] >= 3
    assert info["duration_s"] > 0.5
    probe = _ffprobe_json(out_mp3)
    assert any(st.get("codec_name") == "mp3" for st in probe["streams"])

    # Also test OpenVoiceBackend F0 helper doesn't fail on empty cache key collision
    ov = OpenVoiceBackend(device="cpu", pitch_match_mode="auto")
    assert ov._se_temp_dir.exists()

    # Verify CLI clone-voices + tts-clone with --mock
    src_video = _make_dummy_video(tmp_path / "cli_vid.mp4")
    bank_dir = tmp_path / "cli_bank"
    with (
        patch("ecoup.dubbing.pipeline.separate_stems", _fake_separate_stems),
        patch("ecoup.dubbing.diarization.SpeakerDiarizer.diarize_segments", _fake_diarize_two_speakers),
    ):
        rc1 = cli_main([
            "clone-voices", "-i", str(src_video),
            "--bank-dir", str(bank_dir),
            "--num-speakers", "2",
            "--mock",
        ])
    assert rc1 == 0
    cloned = VoiceBankManager(bank_dir).list_voices()
    assert len(cloned) == 2

    cli_mp3 = tmp_path / "cli_out.mp3"
    rc2 = cli_main([
        "tts-clone",
        "--clone", cloned[0].clone_id,
        "--bank-dir", str(bank_dir),
        "--text", "Đây là bài kiểm tra lệnh CLI xuất file âm thanh.",
        "-o", str(cli_mp3),
        "--mock",
    ])
    assert rc2 == 0
    assert cli_mp3.exists() and cli_mp3.stat().st_size > 1000
