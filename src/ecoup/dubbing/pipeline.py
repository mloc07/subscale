"""Dubbing pipeline coordinator."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Optional

from .asr import AsrOptions, DubbingSegment, transcribe_dialogue
from .audio import AudioProcessingError, NoAudioError, extract_original_hq, probe_audio_info, separate_stems
from .config import DubbingConfig
from .diarization import DiarizationOptions, SpeakerDiarizer, export_speaker_previews
from .manifest import ProjectManifest
from .translate import DubbingTranslator, TranslatedDubSegment, TranslationConfig
from .voice_profile import VoiceProfile, build_voice_profiles


class DubbingPipeline:
    def __init__(
        self,
        video_path: str | Path,
        config: Optional[DubbingConfig] = None,
        on_progress: Optional[Callable[[str, float], None]] = None,
    ):
        self.video_path = Path(video_path)
        self.config = config or DubbingConfig()
        self.config.validate()
        self.on_progress = on_progress
        self._cancelled = False

        # Determine project directory
        if self.config.project_dir:
            self.project_dir = Path(self.config.project_dir)
        else:
            self.project_dir = self.video_path.parent / f"dub_project_{self.video_path.stem}"
        self.project_dir.mkdir(parents=True, exist_ok=True)

        # Standard project subdirectories
        self.audio_dir = self.project_dir / "audio"
        self.audio_dir.mkdir(parents=True, exist_ok=True)

        self.transcript_dir = self.project_dir / "transcript"
        self.transcript_dir.mkdir(parents=True, exist_ok=True)

        self.voices_dir = self.project_dir / "voices"
        self.voices_dir.mkdir(parents=True, exist_ok=True)

        self.translation_dir = self.project_dir / "translation"
        self.translation_dir.mkdir(parents=True, exist_ok=True)

        self.manifest = ProjectManifest.load_or_create(self.project_dir, str(self.video_path))

    def _progress(self, step: str, frac: float = 0.0) -> None:
        if self.on_progress:
            self.on_progress(step, frac)

    def cancel(self) -> None:
        self._cancelled = True

    def run_s0_intake(self) -> Path:
        """S0: Probe video, check audio stream, and extract original_hq.wav."""
        self._progress("S0", 0.0)
        s0_hash = self.config.get_step_hash("S0")
        target_wav = self.audio_dir / "original_hq.wav"

        if self.manifest.is_step_valid("S0", s0_hash, [target_wav]):
            self._progress("S0", 1.0)
            return target_wav

        t0 = time.time()
        info = probe_audio_info(self.video_path)
        extract_original_hq(self.video_path, target_wav, self.config.audio)
        elapsed = time.time() - t0

        self.manifest.record_step(
            step_name="S0",
            config_hash=s0_hash,
            outputs={"original_hq": str(target_wav.relative_to(self.project_dir))},
            metadata={"elapsed_s": round(elapsed, 2), "audio_info": info},
        )
        self._progress("S0", 1.0)
        return target_wav

    def run_s1_separation(self) -> tuple[Path, Path]:
        """S1: Separate vocals and BGM into dialogue.wav and bgm.wav."""
        self._progress("S1", 0.0)
        s1_hash = self.config.get_step_hash("S1")
        dialogue_wav = self.audio_dir / "dialogue.wav"
        bgm_wav = self.audio_dir / "bgm.wav"

        if self.manifest.is_step_valid("S1", s1_hash, [dialogue_wav, bgm_wav]):
            self._progress("S1", 1.0)
            return dialogue_wav, bgm_wav

        orig_wav = self.run_s0_intake()

        t0 = time.time()
        dialogue_out, bgm_out = separate_stems(orig_wav, self.audio_dir, self.config.demucs)
        elapsed = time.time() - t0

        self.manifest.record_step(
            step_name="S1",
            config_hash=s1_hash,
            outputs={
                "dialogue": str(dialogue_out.relative_to(self.project_dir)),
                "bgm": str(bgm_out.relative_to(self.project_dir)),
            },
            metadata={"elapsed_s": round(elapsed, 2)},
        )
        self._progress("S1", 1.0)
        return dialogue_out, bgm_out

    def run_s2_asr(self) -> list[DubbingSegment]:
        """S2: Speech recognition on dialogue.wav, outputs segments.json."""
        self._progress("S2", 0.0)
        s2_hash = self.config.get_step_hash("S2")
        segments_json = self.transcript_dir / "segments.json"

        if self.manifest.is_step_valid("S2", s2_hash, [segments_json]):
            self._progress("S2", 1.0)
            with open(segments_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            return [DubbingSegment(**item) for item in data]

        dialogue_wav, _ = self.run_s1_separation()

        t0 = time.time()
        asr_opts = AsrOptions(
            model_size=self.config.asr.model_size,
            beam_size=self.config.asr.beam_size,
            vad_filter=self.config.asr.vad_filter,
            language=self.config.asr.language,
            mock=self.config.asr.mock,
        )
        segments, lang, prob = transcribe_dialogue(
            dialogue_wav,
            options=asr_opts,
            should_cancel=lambda: self._cancelled,
        )
        elapsed = time.time() - t0

        with open(segments_json, "w", encoding="utf-8") as f:
            json.dump([asdict(s) for s in segments], f, ensure_ascii=False, indent=2)

        self.manifest.record_step(
            step_name="S2",
            config_hash=s2_hash,
            outputs={"segments": str(segments_json.relative_to(self.project_dir))},
            metadata={
                "elapsed_s": round(elapsed, 2),
                "language": lang,
                "language_prob": round(prob, 2),
                "segments_count": len(segments),
            },
        )
        self._progress("S2", 1.0)
        return segments

    def run_s3_diarization(self) -> list[DubbingSegment]:
        """S3: Speaker diarization & clustering, outputs diarized.json."""
        self._progress("S3", 0.0)
        s3_hash = self.config.get_step_hash("S3")
        diarized_json = self.transcript_dir / "diarized.json"

        if self.manifest.is_step_valid("S3", s3_hash, [diarized_json]):
            self._progress("S3", 1.0)
            with open(diarized_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            return [DubbingSegment(**item) for item in data]

        segments = self.run_s2_asr()
        dialogue_wav = self.audio_dir / "dialogue.wav"

        t0 = time.time()
        diar_opts = DiarizationOptions(
            num_speakers=self.config.diarization.num_speakers,
            min_speakers=self.config.diarization.min_speakers,
            max_speakers=self.config.diarization.max_speakers,
            distance_threshold=self.config.diarization.distance_threshold,
            mock=self.config.diarization.mock,
        )
        diarizer = SpeakerDiarizer(diar_opts)
        diarized_segs = diarizer.diarize_segments(
            dialogue_wav,
            segments,
            should_cancel=lambda: self._cancelled,
        )
        elapsed = time.time() - t0

        with open(diarized_json, "w", encoding="utf-8") as f:
            json.dump([asdict(s) for s in diarized_segs], f, ensure_ascii=False, indent=2)

        previews = export_speaker_previews(dialogue_wav, diarized_segs, self.voices_dir)

        outputs = {"diarized": str(diarized_json.relative_to(self.project_dir))}
        for spk, p in previews.items():
            outputs[f"preview_{spk}"] = str(p.relative_to(self.project_dir))

        unique_speakers = sorted(list(set(s.speaker_id for s in diarized_segs if s.speaker_id)))
        self.manifest.record_step(
            step_name="S3",
            config_hash=s3_hash,
            outputs=outputs,
            metadata={
                "elapsed_s": round(elapsed, 2),
                "unique_speakers": unique_speakers,
                "overlaps_count": sum(1 for s in diarized_segs if s.is_overlap),
            },
        )
        self._progress("S3", 1.0)
        return diarized_segs

    def run_s4_voice_profiles(self) -> dict[str, VoiceProfile]:
        """S4: Voice profiling & reference audio extraction for each speaker."""
        self._progress("S4", 0.0)
        s4_hash = self.config.get_step_hash("S4")

        # Check if profiles already exist
        spk_profiles = {}
        all_exist = True
        for spk_dir in self.voices_dir.glob("spk_*"):
            p_json = spk_dir / "profile.json"
            ref_wav = spk_dir / "ref_clean.wav"
            if p_json.exists() and ref_wav.exists():
                with open(p_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
                spk_profiles[spk_dir.name] = VoiceProfile(**data)
            else:
                all_exist = False

        if all_exist and spk_profiles and self.manifest.is_step_valid("S4", s4_hash):
            self._progress("S4", 1.0)
            return spk_profiles

        diarized_segs = self.run_s3_diarization()
        dialogue_wav = self.audio_dir / "dialogue.wav"

        t0 = time.time()
        profiles = build_voice_profiles(dialogue_wav, diarized_segs, self.voices_dir, target_sr=24000)
        elapsed = time.time() - t0

        outputs = {}
        for spk, vp in profiles.items():
            outputs[f"ref_{spk}"] = vp.reference_wav

        self.manifest.record_step(
            step_name="S4",
            config_hash=s4_hash,
            outputs=outputs,
            metadata={
                "elapsed_s": round(elapsed, 2),
                "profiles_summary": {k: {"quality": v.quality_score, "is_good": v.is_good_reference} for k, v in profiles.items()},
            },
        )
        self._progress("S4", 1.0)
        return profiles

    def run_s5_translation(self, context_notes: str = "") -> list[TranslatedDubSegment]:
        """S5: Translate dialogue to Vietnamese with duration budget constraint."""
        self._progress("S5", 0.0)
        s5_hash = self.config.get_step_hash("S5")
        translated_json = self.translation_dir / "translated.json"

        if self.manifest.is_step_valid("S5", s5_hash, [translated_json]):
            self._progress("S5", 1.0)
            with open(translated_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            return [TranslatedDubSegment(**item) for item in data]

        diarized_segs = self.run_s3_diarization()

        t0 = time.time()
        trans_cfg = TranslationConfig(
            model=self.config.translation.model,
            syllables_per_second=self.config.translation.syllables_per_second,
            max_rewrite_attempts=self.config.translation.max_rewrite_attempts,
            temperature=self.config.translation.temperature,
            mock=self.config.translation.mock,
        )
        translator = DubbingTranslator(trans_cfg)
        translated_segs = translator.translate_segments(
            diarized_segs,
            context_notes=context_notes,
            should_cancel=lambda: self._cancelled,
        )
        elapsed = time.time() - t0

        with open(translated_json, "w", encoding="utf-8") as f:
            json.dump([asdict(s) for s in translated_segs], f, ensure_ascii=False, indent=2)

        overflow_count = sum(1 for s in translated_segs if s.overflow_flag)
        rewritten_count = sum(1 for s in translated_segs if s.rewritten_times > 0)

        self.manifest.record_step(
            step_name="S5",
            config_hash=s5_hash,
            outputs={"translated": str(translated_json.relative_to(self.project_dir))},
            metadata={
                "elapsed_s": round(elapsed, 2),
                "total_translated": len(translated_segs),
                "overflow_count": overflow_count,
                "rewritten_count": rewritten_count,
            },
        )
        self._progress("S5", 1.0)
        return translated_segs
