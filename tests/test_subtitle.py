"""Unit tests for the subtitle module (no real API / model calls)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from ecoup.subtitle.asr import AsrResult
from ecoup.subtitle.burn import escape_filter_value
from ecoup.subtitle.config import BurnConfig, SegmentConfig, SubtitleConfig, TranslateConfig
from ecoup.subtitle.pipeline import SubtitlePipeline
from ecoup.subtitle.segments import Segment, TranslatedSegment, normalize_segments, wrap_lines
from ecoup.subtitle.srt import build_ass, format_ts, parse_ts, read_srt, validate_timeline, write_srt
from ecoup.subtitle.translate.gemini import (
    BlockedError, CountMismatchError, GeminiTranslator, RetryableError, TranslationError,
)

HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


# ----------------------------------------------------------------- segments
def test_normalize_empty_and_blank():
    assert normalize_segments([], SegmentConfig()) == []
    assert normalize_segments([Segment(0, 1, "   "), Segment(1, 2, "")], SegmentConfig()) == []


def test_normalize_merges_short_and_removes_overlap():
    cfg = SegmentConfig()
    segs = [Segment(0.0, 0.3, "Hi"), Segment(0.4, 2.0, "there friend"), Segment(1.9, 3.0, "overlap")]
    out = normalize_segments(segs, cfg)
    assert out[0].text == "Hi there friend"
    validate_timeline([(s.start, s.end) for s in out])


def test_normalize_splits_long_segment():
    cfg = SegmentConfig(max_duration=4.0)
    seg = Segment(0.0, 12.0, "One sentence here. Another sentence follows. A third one ends it.")
    out = normalize_segments([seg], cfg)
    assert len(out) >= 3
    assert out[0].start == 0.0 and abs(out[-1].end - 12.0) < 1e-6 or out[-1].end <= 12.0
    validate_timeline([(s.start, s.end) for s in out])


def test_normalize_fixes_inverted_timing():
    out = normalize_segments([Segment(5.0, 5.0, "zero length")], SegmentConfig())
    assert out and out[0].end > out[0].start


def test_wrap_lines_vietnamese_and_cjk():
    vi = wrap_lines("Xin chào các bạn, hôm nay chúng ta sẽ học một bài rất thú vị và dài", 20, 2)
    assert vi.count("\n") <= 1 and "Xin chào" in vi
    cjk = wrap_lines("这是一个很长很长的中文字幕需要被自动换行处理才可以显示", 10, 2)
    assert cjk.count("\n") <= 1 and cjk.replace("\n", "").startswith("这是一个")


# --------------------------------------------------------------------- srt
def test_timestamp_roundtrip():
    for t in (0.0, 1.5, 59.999, 3661.25):
        assert abs(parse_ts(format_ts(t)) - t) < 1e-3


def test_srt_write_read_unicode(tmp_path):
    cues = [(0.0, 1.0, "Xin chào thế giới – dấu tiếng Việt"), (1.2, 2.0, "日本語\nline 2")]
    p = write_srt(tmp_path / "a.srt", cues)
    assert p.read_bytes().decode("utf-8")  # valid UTF-8
    back = read_srt(p)
    assert [c[2] for c in back] == [c[2] for c in cues]


def test_srt_rejects_overlap_and_inverted(tmp_path):
    with pytest.raises(ValueError):
        write_srt(tmp_path / "x.srt", [(0, 2, "a"), (1, 3, "b")])
    with pytest.raises(ValueError):
        write_srt(tmp_path / "y.srt", [(2, 1, "a")])


def test_srt_empty_cue_list(tmp_path):
    p = write_srt(tmp_path / "e.srt", [])
    assert read_srt(p) == []


def test_ass_scales_with_resolution_and_escapes():
    small = build_ass([(0, 1, "a\nb {x}")], 406, 720, BurnConfig())
    big = build_ass([(0, 1, "a")], 1620, 1080, BurnConfig())
    assert "PlayResX: 406" in small and "a\\Nb (x)" in small
    fs_small = int(small.split("Style: Default,")[1].split(",")[1])
    fs_big = int(big.split("Style: Default,")[1].split(",")[1])
    assert fs_small < fs_big


def test_escape_filter_value():
    assert escape_filter_value(r"C:\Users\a b\fonts") == r"C\:/Users/a b/fonts"


# ------------------------------------------------------------------ gemini
def _tr(tmp_path=None, **kw):
    cfg = TranslateConfig(cache_dir=str(tmp_path) if tmp_path else None, **kw)
    sleeps: list[float] = []
    return GeminiTranslator(cfg, client=object(), sleep=sleeps.append), sleeps


def _segs(n):
    return [Segment(i, i + 0.9, f"line {i}", "en") for i in range(n)]


def _resp(payload, finish="STOP", block=None):
    return SimpleNamespace(
        text=json.dumps(payload),
        prompt_feedback=SimpleNamespace(block_reason=block) if block else None,
        candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name=finish))],
    )


def test_parse_response_ok_and_mismatch():
    ok = GeminiTranslator._parse_response(_resp([{"id": 1, "text": "b"}, {"id": 0, "text": "a"}]), 2)
    assert ok == ["a", "b"]
    with pytest.raises(CountMismatchError):
        GeminiTranslator._parse_response(_resp([{"id": 0, "text": "a"}]), 2)
    with pytest.raises(BlockedError):
        GeminiTranslator._parse_response(_resp([], finish="SAFETY"), 1)
    with pytest.raises(BlockedError):
        GeminiTranslator._parse_response(_resp([], block="PROHIBITED_CONTENT"), 1)


def test_translate_retries_backoff_then_succeeds(monkeypatch):
    tr, sleeps = _tr()
    calls = {"n": 0}

    def fake(batch, before, after, lang):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RetryableError("429")
        return [f"vi {s.text}" for s in batch]

    monkeypatch.setattr(tr, "_call_api", fake)
    out = tr.translate(_segs(2))
    assert [o.text for o in out] == ["vi line 0", "vi line 1"]
    assert len(sleeps) == 2 and sleeps[1] > sleeps[0]
    assert tr.stats["retries"] == 2


def test_translate_count_mismatch_splits_batch(monkeypatch):
    tr, _ = _tr()
    sizes = []

    def fake(batch, before, after, lang):
        sizes.append(len(batch))
        if len(batch) > 2:
            raise CountMismatchError("bad")
        return [f"vi {s.text}" for s in batch]

    monkeypatch.setattr(tr, "_call_api", fake)
    out = tr.translate(_segs(5))
    assert [o.text for o in out] == [f"vi line {i}" for i in range(5)]
    assert sizes[0] == 5 and max(sizes[1:]) <= 3
    assert not any(o.failed for o in out)


def test_translate_blocked_keeps_original_and_flags(monkeypatch):
    tr, _ = _tr()

    def fake(batch, before, after, lang):
        if any("line 1" == s.text for s in batch):
            raise BlockedError("SAFETY")
        return [f"vi {s.text}" for s in batch]

    monkeypatch.setattr(tr, "_call_api", fake)
    out = tr.translate(_segs(3))
    assert [o.failed for o in out] == [False, True, False]
    assert out[1].text == "line 1" and "blocked" in out[1].note
    assert tr.stats["failed_segments"] == 1 and tr.stats["blocked_batches"] >= 1


def test_translate_blocked_uses_fallback(monkeypatch):
    class Fallback:
        def translate(self, batch, before=(), after=(), source_language=None):
            return [TranslatedSegment(s.start, s.end, s.text, "fb " + s.text, s.language) for s in batch]

    cfg = TranslateConfig()
    tr = GeminiTranslator(cfg, client=object(), sleep=lambda _: None, fallback=Fallback())
    monkeypatch.setattr(tr, "_call_api", lambda *a: (_ for _ in ()).throw(BlockedError("x")))
    out = tr.translate(_segs(1))
    assert out[0].text == "fb line 0" and not out[0].failed and tr.stats["fallback_segments"] == 1


def test_translate_cache_hit_skips_api(monkeypatch, tmp_path):
    tr, _ = _tr(tmp_path)
    n = {"c": 0}

    def fake(batch, before, after, lang):
        n["c"] += 1
        return [f"vi {s.text}" for s in batch]

    monkeypatch.setattr(tr, "_call_api", fake)
    tr.translate(_segs(3))
    tr2, _ = _tr(tmp_path)
    monkeypatch.setattr(tr2, "_call_api", lambda *a: (_ for _ in ()).throw(AssertionError("should be cached")))
    out = tr2.translate(_segs(3))
    assert n["c"] == 1 and tr2.stats["cache_hits"] == 1 and out[2].text == "vi line 2"


def test_unavailable_model_switches_to_fallback_then_errors(monkeypatch, tmp_path):
    cfg = TranslateConfig(max_retries=1, fallback_models=["fb-model"], cache_dir=str(tmp_path))
    tr = GeminiTranslator(cfg, client=object(), sleep=lambda _: None)
    seen = []

    def fake(batch, before, after, lang):
        seen.append(tr._model)
        if tr._model == cfg.model:
            raise RetryableError("503")
        return [f"vi {s.text}" for s in batch]

    monkeypatch.setattr(tr, "_call_api", fake)
    out = tr.translate(_segs(2))
    assert [o.text for o in out] == ["vi line 0", "vi line 1"]
    assert seen == [cfg.model, cfg.model, "fb-model"] and tr.stats["model_switches"] == 1

    tr2 = GeminiTranslator(TranslateConfig(max_retries=0, fallback_models=["fb"]), client=object(), sleep=lambda _: None)
    monkeypatch.setattr(tr2, "_call_api", lambda *a: (_ for _ in ()).throw(RetryableError("503")))
    with pytest.raises(TranslationError, match="unavailable"):
        tr2.translate(_segs(3))


def test_auth_error_is_fatal(monkeypatch):
    tr, _ = _tr()
    monkeypatch.setattr(tr, "_call_api", lambda *a: (_ for _ in ()).throw(TranslationError("401")))
    with pytest.raises(TranslationError):
        tr.translate(_segs(2))


def test_api_key_never_in_error(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    orig_exists = Path.exists
    monkeypatch.setattr(Path, "exists", lambda self: False if ".gemini_key" in str(self) else orig_exists(self))
    tr = GeminiTranslator(TranslateConfig(), sleep=lambda _: None)
    with pytest.raises(TranslationError) as ei:
        tr.translate(_segs(1))
    assert "GEMINI_API_KEY" in str(ei.value)


def test_config_validation():
    with pytest.raises(ValueError):
        TranslateConfig(batch_size=0).validate()
    with pytest.raises(ValueError):
        SegmentConfig(min_duration=5, max_duration=2).validate()
    SubtitleConfig().validate()


# ---------------------------------------------------------------- pipeline
class FakeAsr:
    def __init__(self, segs, lang="en"):
        self.segs, self.lang = segs, lang

    def transcribe(self, wav, cfg, on_segment=None, should_cancel=None):
        return AsrResult(segments=self.segs, language=self.lang, language_probability=0.99)


class FakeTranslator:
    stats = {"requests": 0}

    def translate(self, batch, before=(), after=(), source_language=None):
        self.stats["requests"] += 1
        return [TranslatedSegment(s.start, s.end, s.text, "VI: " + s.text, s.language) for s in batch]


@pytest.fixture()
def tiny_video(tmp_path):
    if not HAVE_FFMPEG:
        pytest.skip("ffmpeg not available")
    p = tmp_path / "in video (1).mp4"  # spaces + parentheses on purpose
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=blue:s=360x640:d=4:r=10",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=4", "-shortest", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-c:a", "aac", str(p)], check=True)
    return p


@pytest.fixture()
def silent_video(tmp_path):
    if not HAVE_FFMPEG:
        pytest.skip("ffmpeg not available")
    p = tmp_path / "silent.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=red:s=160x160:d=1:r=5",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(p)], check=True)
    return p


def test_pipeline_end_to_end_with_fakes(tiny_video, tmp_path):
    asr = FakeAsr([Segment(0.2, 1.8, "Hello there my friend", "en"), Segment(2.0, 3.5, "How are you today", "en")])
    cfg = SubtitleConfig()
    cfg.burn.font_name = "Arial"
    pipe = SubtitlePipeline(cfg, asr=asr, translator=FakeTranslator())
    out = tmp_path / "out dir" / "res.mp4"
    rep = pipe.run(tiny_video, output=out)
    assert out.exists() and rep.detected_language == "en" and rep.cues == 2
    assert not rep.skipped_translation
    assert rep.outputs["srt_original"].endswith(".en.srt") and rep.outputs["srt_translated"].endswith(".vi.srt")
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(out)],
                           capture_output=True, text=True).stdout
    assert "video" in probe and "audio" in probe  # audio kept
    assert read_srt(rep.outputs["srt_translated"])[0][2].startswith("VI:")


def test_pipeline_skips_translation_for_vietnamese(tiny_video, tmp_path):
    asr = FakeAsr([Segment(0.2, 1.8, "Xin chào các bạn", "vi")], lang="vi")
    cfg = SubtitleConfig()
    cfg.burn_in = False
    rep = SubtitlePipeline(cfg, asr=asr, translator=None).run(tiny_video, out_dir=tmp_path / "o")
    assert rep.skipped_translation and "video" not in rep.outputs


def test_pipeline_no_audio_raises_clear_error(silent_video, tmp_path):
    from ecoup.subtitle.audio import NoAudioError
    with pytest.raises(NoAudioError):
        SubtitlePipeline(SubtitleConfig(), asr=FakeAsr([]), translator=FakeTranslator()).run(silent_video, out_dir=tmp_path)


def test_pipeline_empty_asr_and_cancel(tiny_video, tmp_path):
    cfg = SubtitleConfig()
    cfg.burn_in = False
    rep = SubtitlePipeline(cfg, asr=FakeAsr([]), translator=FakeTranslator()).run(tiny_video, out_dir=tmp_path / "e")
    assert rep.cues == 0 and any("no speech" in w for w in rep.warnings)

    class CancellingAsr(FakeAsr):
        def transcribe(self, wav, cfg, on_segment=None, should_cancel=None):
            pipe.cancel()
            return super().transcribe(wav, cfg)

    pipe = SubtitlePipeline(cfg, asr=CancellingAsr([Segment(0, 1, "x y z", "en")]), translator=FakeTranslator())
    assert pipe.run(tiny_video, out_dir=tmp_path / "c").cancelled


def test_pipeline_with_custom_srt_bypasses_asr(tiny_video, tmp_path):
    # Tests preview/custom SRT execution (bypasses ASR and audio extraction)
    srt_file = tmp_path / "manual.srt"
    write_srt(srt_file, [(0.5, 2.5, "Custom manual line 1"), (3.0, 5.0, "Custom manual line 2")])
    cfg = SubtitleConfig()
    cfg.burn_in = False
    pipe = SubtitlePipeline(cfg, asr=None, translator=FakeTranslator())
    rep = pipe.run(tiny_video, out_dir=tmp_path / "custom_out", custom_srt=srt_file)
    assert rep.cues == 2
    assert "audio" not in rep.timings_s and "asr" not in rep.timings_s
    assert "srt_translated" in rep.outputs
    tr_srt = read_srt(rep.outputs["srt_translated"])
    assert tr_srt[0][2].startswith("VI: Custom manual line 1")


def test_pipeline_transcribe_only(tiny_video, tmp_path):
    # Tests transcribe_only for human review
    asr = FakeAsr([Segment(0.5, 2.0, "First spoken words", "en")], lang="en")
    cfg = SubtitleConfig()
    pipe = SubtitlePipeline(cfg, asr=asr, translator=FakeTranslator())
    srt_path, srt_text, rep = pipe.transcribe_only(tiny_video, out_dir=tmp_path / "review")
    assert srt_path.exists()
    assert "First spoken words" in srt_text
    assert rep.cues == 1


def test_lyric_snapping():
    from ecoup.subtitle.snapping import parse_raw_lyrics, snap_lyrics_to_segments
    raw_lyric = """[Verse 1]
Baby, come near me now
You see this your love, don't tie me down
Call me your prisoner"""
    lines = parse_raw_lyrics(raw_lyric)
    assert len(lines) == 3
    assert lines[0] == "Baby, come near me now"

    # Whisper slightly misheard
    whisper_segs = [
        Segment(0.0, 2.0, "Baby come near me now"),
        Segment(2.2, 4.5, "You see this love dont tie me down"),
        Segment(4.8, 6.0, "Call me your prisoner"),
    ]
    snapped = snap_lyrics_to_segments(whisper_segs, raw_lyric)
    assert snapped[1].text == "You see this your love, don't tie me down"


def test_pipeline_with_demucs_mock(tiny_video, tmp_path, monkeypatch):
    called = []
    def fake_extract_vocals(media_path, out_wav_path, sample_rate=16000):
        called.append(media_path)
        p = Path(out_wav_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"RIFFdummywavdata")
        return p

    import ecoup.subtitle.audio
    monkeypatch.setattr(ecoup.subtitle.audio, "extract_vocals", fake_extract_vocals)

    cfg = SubtitleConfig()
    cfg.asr.use_demucs = True
    cfg.burn_in = False
    asr = FakeAsr([Segment(0.5, 2.0, "Vocal line", "en")], lang="en")
    pipe = SubtitlePipeline(cfg, asr=asr, translator=FakeTranslator())
    rep = pipe.run(tiny_video, out_dir=tmp_path / "demucs_test")
    assert len(called) == 1
    assert "audio_hq.wav" in str(called[0])
    assert rep.demucs_applied is True
    assert rep.cues == 1


def test_pipeline_demucs_fallback_on_failure(tiny_video, tmp_path, monkeypatch):
    def failing_extract_vocals(media_path, out_wav_path, sample_rate=16000):
        raise RuntimeError("CUDA Out of Memory in Demucs")

    import ecoup.subtitle.audio
    monkeypatch.setattr(ecoup.subtitle.audio, "extract_vocals", failing_extract_vocals)

    cfg = SubtitleConfig()
    cfg.asr.use_demucs = True
    cfg.burn_in = False
    asr = FakeAsr([Segment(0.5, 2.0, "Fallback line", "en")], lang="en")
    pipe = SubtitlePipeline(cfg, asr=asr, translator=FakeTranslator())
    rep = pipe.run(tiny_video, out_dir=tmp_path / "demucs_fail_test")
    assert rep.demucs_applied is False
    assert any("Demucs vocal separation warning" in w for w in rep.warnings)
    assert rep.cues == 1



