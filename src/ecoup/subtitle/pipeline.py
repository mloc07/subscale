"""SubtitlePipeline: extract audio -> ASR -> normalise -> translate -> .srt -> burn-in."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

from .asr import AsrBackend, FasterWhisperBackend
from .audio import NoAudioError, extract_audio
from .burn import burn_subtitles, make_ass_for_video
from .config import SubtitleConfig
from .report import SubtitleReport
from .segments import Segment, TranslatedSegment, normalize_segments
from .srt import read_srt, segments_to_cues, translated_to_cues, write_srt
from .translate.base import TranslatorBackend

STEPS = ("audio", "asr", "normalize", "translate", "srt", "burn")


class SubtitlePipeline:
    def __init__(
        self,
        config: Optional[SubtitleConfig] = None,
        asr: Optional[AsrBackend] = None,
        translator: Optional[TranslatorBackend] = None,
        on_progress: Optional[Callable[[str, float], None]] = None,
        encode=None,
    ) -> None:
        self.config = config or SubtitleConfig()
        self.config.validate()
        self.asr = asr or FasterWhisperBackend()
        self._translator = translator
        self.on_progress = on_progress
        self.encode = encode
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def _progress(self, step: str, frac: float = 0.0) -> None:
        if self.on_progress:
            self.on_progress(step, frac)

    def _get_translator(self) -> TranslatorBackend:
        if self._translator is None:
            from .translate.gemini import GeminiTranslator
            self._translator = GeminiTranslator(self.config.translate, should_cancel=lambda: self._cancelled)
        return self._translator

    def transcribe_only(
        self,
        video: str | Path,
        out_dir: Optional[str | Path] = None,
    ) -> tuple[Path, str, SubtitleReport]:
        """Run only Audio Extraction + Whisper ASR, producing original SRT for preview/editing."""
        cfg = self.config
        video = Path(video)
        out_dir = Path(out_dir) if out_dir else video.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = video.stem
        work = Path(cfg.work_dir) if cfg.work_dir else out_dir / f".sub_work_{stem}"
        work.mkdir(parents=True, exist_ok=True)

        rep = SubtitleReport(video=str(video))
        t_all = time.time()

        # 1. audio
        self._progress("audio", 0.0)
        t = time.time()
        try:
            if cfg.asr.use_demucs:
                try:
                    from .audio import extract_vocals
                    # Extract 44.1 kHz stereo audio for optimal Demucs separation quality
                    audio_hq = extract_audio(video, work / "audio_hq.wav", sample_rate=44100, channels=2)
                    wav = extract_vocals(audio_hq, work / "vocals.wav", sample_rate=16000)
                    rep.demucs_applied = True
                except NoAudioError:
                    raise
                except Exception as exc:
                    rep.demucs_applied = False
                    rep.warnings.append(f"Demucs vocal separation warning: {exc}. Fallback to original audio.")
                    wav = extract_audio(video, work / "audio.wav", sample_rate=16000, channels=1)
            else:
                wav = extract_audio(video, work / "audio.wav", sample_rate=16000, channels=1)
        except NoAudioError as exc:
            rep.warnings.append(str(exc))
            raise
        rep.timings_s["audio"] = round(time.time() - t, 2)
        if self._cancelled:
            return Path(""), "", self._cancel(rep)

        # 2. ASR
        self._progress("asr", 0.0)
        t = time.time()
        res = self.asr.transcribe(str(wav), cfg.asr, on_segment=None, should_cancel=lambda: self._cancelled)
        rep.timings_s["asr"] = round(time.time() - t, 2)
        rep.detected_language = res.language
        rep.language_probability = res.language_probability
        rep.asr_peak_vram_gb = res.peak_vram_gb
        rep.asr_segments = len(res.segments)
        if self._cancelled:
            return Path(""), "", self._cancel(rep)
        if not res.segments:
            rep.warnings.append("ASR produced no speech segments.")

        # 3. normalise + write original SRT
        segs = normalize_segments(res.segments, cfg.segment)
        for s in segs:
            if not s.language:
                s.language = res.language
        rep.cues = len(segs)
        orig_srt = write_srt(
            out_dir / f"{stem}.{res.language or 'src'}.srt",
            segments_to_cues(segs, cfg.segment.max_chars_per_line, cfg.segment.max_lines),
        )
        rep.outputs["srt_original"] = str(orig_srt)
        rep.timings_s["total"] = round(time.time() - t_all, 2)
        self._progress("asr", 1.0)
        srt_content = orig_srt.read_text(encoding="utf-8")
        return orig_srt, srt_content, rep

    def run(
        self,
        video: str | Path,
        output: Optional[str | Path] = None,
        out_dir: Optional[str | Path] = None,
        custom_srt: Optional[str | Path] = None,
    ) -> SubtitleReport:
        cfg = self.config
        video = Path(video)
        out_dir = Path(out_dir) if out_dir else (Path(output).parent if output else video.parent)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = video.stem
        output = Path(output) if output else out_dir / f"{stem}_sub.mp4"
        work = Path(cfg.work_dir) if cfg.work_dir else out_dir / f".sub_work_{stem}"
        work.mkdir(parents=True, exist_ok=True)

        rep = SubtitleReport(video=str(video))
        t_all = time.time()

        if custom_srt:
            # Custom / pre-reviewed SRT provided by user: bypass Audio & ASR
            custom_srt_path = Path(custom_srt)
            if not custom_srt_path.is_file():
                # Provided as raw SRT text content
                tmp_srt = work / "custom_input.srt"
                tmp_srt.write_text(str(custom_srt), encoding="utf-8")
                custom_srt_path = tmp_srt
            cues_in = read_srt(custom_srt_path)
            res_lang = cfg.asr.language or "src"
            rep.detected_language = res_lang
            raw_segs = [Segment(start=c[0], end=c[1], text=c[2], language=res_lang) for c in cues_in]
            segs = normalize_segments(raw_segs, cfg.segment)
            for s in segs:
                if not s.language:
                    s.language = res_lang
            orig_srt = custom_srt_path
            rep.outputs["srt_original"] = str(orig_srt)
            rep.cues = len(segs)
        else:
            # 1. audio
            self._progress("audio", 0.0)
            t = time.time()
            try:
                if cfg.asr.use_demucs:
                    try:
                        from .audio import extract_vocals
                        # Extract 44.1 kHz stereo audio for optimal Demucs separation quality
                        audio_hq = extract_audio(video, work / "audio_hq.wav", sample_rate=44100, channels=2)
                        wav = extract_vocals(audio_hq, work / "vocals.wav", sample_rate=16000)
                        rep.demucs_applied = True
                    except NoAudioError:
                        raise
                    except Exception as exc:
                        rep.demucs_applied = False
                        rep.warnings.append(f"Demucs vocal separation warning: {exc}. Fallback to original audio.")
                        wav = extract_audio(video, work / "audio.wav", sample_rate=16000, channels=1)
                else:
                    wav = extract_audio(video, work / "audio.wav", sample_rate=16000, channels=1)
            except NoAudioError as exc:
                rep.warnings.append(str(exc))
                raise
            rep.timings_s["audio"] = round(time.time() - t, 2)
            if self._cancelled:
                return self._cancel(rep)

            # 2. ASR
            self._progress("asr", 0.0)
            t = time.time()
            res = self.asr.transcribe(str(wav), cfg.asr, on_segment=None, should_cancel=lambda: self._cancelled)
            rep.timings_s["asr"] = round(time.time() - t, 2)
            rep.detected_language = res.language
            rep.language_probability = res.language_probability
            rep.asr_peak_vram_gb = res.peak_vram_gb
            rep.asr_segments = len(res.segments)
            if self._cancelled:
                return self._cancel(rep)
            if not res.segments:
                rep.warnings.append("ASR produced no speech segments.")

            # 3. normalise + original SRT
            segs = normalize_segments(res.segments, cfg.segment)
            # FIX H4: Preserve individual segment language instead of overwriting all
            for s in segs:
                if not s.language:
                    s.language = res.language
            rep.cues = len(segs)
            orig_srt = write_srt(out_dir / f"{stem}.{res.language or 'src'}.srt",
                                 segments_to_cues(segs, cfg.segment.max_chars_per_line, cfg.segment.max_lines))
            rep.outputs["srt_original"] = str(orig_srt)
            res_lang = res.language

        self._progress("normalize", 1.0)

        # 4. translate
        target = cfg.translate.target_language
        translated: list[TranslatedSegment]
        if (res_lang or "").lower().split("-")[0] == target.lower().split("-")[0]:
            rep.skipped_translation = True
            translated = [TranslatedSegment(s.start, s.end, s.text, s.text, s.language) for s in segs]
        else:
            t = time.time()
            translated = self._translate_all(segs, res_lang)
            rep.timings_s["translate"] = round(time.time() - t, 2)
            tr = self._translator
            rep.translate_stats = dict(getattr(tr, "stats", {}) or {})
        if self._cancelled:
            return self._cancel(rep)
        rep.translated_failed = sum(1 for s in translated if s.failed)
        if rep.translated_failed:
            rep.warnings.append(
                f"{rep.translated_failed} segment(s) kept in original language (translation failed/blocked)."
            )

        # 5. translated SRT (re-normalised for target-language reading speed)
        cues = translated_to_cues(translated, cfg.segment.max_chars_per_line, cfg.segment.max_lines)
        tr_srt = write_srt(out_dir / f"{stem}.{target}.srt", cues)
        rep.outputs["srt_translated"] = str(tr_srt)
        self._progress("srt", 1.0)

        # 6. burn-in
        if cfg.burn_in:
            self._progress("burn", 0.0)
            t = time.time()
            orig_cues = segments_to_cues(segs, cfg.segment.max_chars_per_line, cfg.segment.max_lines) if cfg.burn.show_original else None
            ass_text = make_ass_for_video(video, cues, cfg.burn, original_cues=orig_cues)
            ass_path = out_dir / f"{stem}.{target}.ass"
            ass_path.write_text(ass_text, encoding="utf-8")
            rep.outputs["ass"] = str(ass_path)
            burn_subtitles(video, ass_path, output, encode=self.encode, fonts_dir=cfg.burn.fonts_dir)
            rep.outputs["video"] = str(output)
            rep.timings_s["burn"] = round(time.time() - t, 2)
            self._progress("burn", 1.0)

        rep.timings_s["total"] = round(time.time() - t_all, 2)
        rep.save(out_dir / f"{stem}.subtitle_report.json")
        return rep

    def _cancel(self, rep: SubtitleReport) -> SubtitleReport:
        rep.cancelled = True
        return rep

    def _translate_all(self, segs: list[Segment], language: Optional[str]) -> list[TranslatedSegment]:
        tc = self.config.translate
        translator = self._get_translator()
        out: list[TranslatedSegment] = []
        n = len(segs)
        for i in range(0, n, tc.batch_size):
            if self._cancelled:
                break
            batch = segs[i : i + tc.batch_size]
            before = segs[max(0, i - tc.context_segments) : i]
            after = segs[i + len(batch) : i + len(batch) + tc.context_segments]
            part = translator.translate(batch, before, after, language)
            if len(part) != len(batch):
                raise RuntimeError("Translator returned wrong number of segments (backend contract violated).")
            out.extend(part)
            self._progress("translate", min(1.0, (i + len(batch)) / max(1, n)))
        return out
