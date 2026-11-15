"""Complete end-to-end AI Dubbing Pipeline."""

from __future__ import annotations

import json
import time
import wave
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .asr import AsrOptions, DubbingSegment, load_wav_float32, transcribe_dialogue
from .audio import AudioProcessingError, NoAudioError, extract_original_hq, probe_audio_info, separate_stems
from .config import DubbingConfig
from .diarization import DiarizationOptions, SpeakerDiarizer, _write_wav_pcm16, export_speaker_previews
from .manifest import ProjectManifest
from .mixing import mix_audio_tracks
from .mux import export_srt_file, mux_video_with_audio
from .snapping import align_audio_to_target_duration
from .translate import DubbingTranslator, TranslatedDubSegment, TranslationConfig
from .tts import (
    F5VoiceBackend,
    GTTSVoiceBackend,
    MockVoiceBackend,
    OpenVoiceBackend,
    VoiceBackend,
    estimate_median_f0_hz,
)
from .voice_bank import SPECIAL_ORIGINAL, SPECIAL_SELF, VoiceBankManager
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

        if self.config.project_dir:
            self.project_dir = Path(self.config.project_dir)
        else:
            self.project_dir = self.video_path.parent / f"dub_project_{self.video_path.stem}"
        self.project_dir.mkdir(parents=True, exist_ok=True)

        self.audio_dir = self.project_dir / "audio"
        self.audio_dir.mkdir(parents=True, exist_ok=True)

        self.transcript_dir = self.project_dir / "transcript"
        self.transcript_dir.mkdir(parents=True, exist_ok=True)

        self.voices_dir = self.project_dir / "voices"
        self.voices_dir.mkdir(parents=True, exist_ok=True)

        self.translation_dir = self.project_dir / "translation"
        self.translation_dir.mkdir(parents=True, exist_ok=True)

        self.tts_dir = self.project_dir / "tts"
        self.tts_dir.mkdir(parents=True, exist_ok=True)

        self.mix_dir = self.project_dir / "mix"
        self.mix_dir.mkdir(parents=True, exist_ok=True)

        self.out_dir = self.project_dir / "out"
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.manifest = ProjectManifest.load_or_create(self.project_dir, str(self.video_path))

    def _progress(self, step: str, frac: float = 0.0) -> None:
        if self.on_progress:
            self.on_progress(step, frac)

    def cancel(self) -> None:
        self._cancelled = True

    def run_s0_intake(self) -> Path:
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
        self._progress("S4", 0.0)
        s4_hash = self.config.get_step_hash("S4")

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
            metadata={"elapsed_s": round(elapsed, 2)},
        )
        self._progress("S4", 1.0)
        return profiles

    def analyze_target_speakers(self) -> list[dict]:
        """Run S0-S4 and return detected speakers ordered by first appearance with previews & F0."""
        diarized_segs = self.run_s3_diarization()
        profiles = self.run_s4_voice_profiles()

        segs_by_spk: dict[str, list[DubbingSegment]] = {}
        first_seen: dict[str, float] = {}
        for s in diarized_segs:
            spk = s.speaker_id or "spk_0"
            segs_by_spk.setdefault(spk, []).append(s)
            if spk not in first_seen or s.start < first_seen[spk]:
                first_seen[spk] = float(s.start)

        all_spks = set(segs_by_spk.keys()) | set(profiles.keys())
        ordered_spks = sorted(all_spks, key=lambda spk: (first_seen.get(spk, 1e9), spk))

        results: list[dict] = []
        for idx, spk_id in enumerate(ordered_spks, start=1):
            prof = profiles.get(spk_id)
            segs = segs_by_spk.get(spk_id, [])
            preview_wav = self.voices_dir / spk_id / "preview.wav"
            ref_wav = (self.project_dir / prof.reference_wav) if prof else (self.voices_dir / spk_id / "ref_clean.wav")

            f0_hz = 0.0
            sample_wav = ref_wav if ref_wav.exists() else preview_wav
            if sample_wav.exists():
                try:
                    wav_audio = load_wav_float32(sample_wav, target_sr=16000)
                    f0_hz = estimate_median_f0_hz(wav_audio, 16000)
                except Exception:
                    f0_hz = 0.0

            texts = [s.text.strip() for s in segs if (s.text or "").strip()]
            sample_text = " | ".join(texts[:2])
            if not sample_text and prof:
                sample_text = prof.reference_text or ""
            if len(sample_text) > 160:
                sample_text = sample_text[:157] + "..."

            speaking_s = (
                prof.speaking_duration_s
                if prof and prof.speaking_duration_s > 0
                else sum(s.duration for s in segs)
            )
            results.append({
                "index": idx,
                "speaker_id": spk_id,
                "label": f"Giọng {idx} ({spk_id})",
                "preview_wav": str(preview_wav) if preview_wav.exists() else (str(ref_wav) if ref_wav.exists() else ""),
                "ref_wav": str(ref_wav) if ref_wav.exists() else "",
                "sample_text": sample_text,
                "speaking_duration_s": round(float(speaking_s), 2),
                "total_segments": len(segs) or (prof.total_segments if prof else 0),
                "estimated_f0_hz": round(float(f0_hz), 1),
                "first_start": round(first_seen.get(spk_id, 0.0), 2),
            })
        return results

    def run_s5_translation(self, context_notes: str = "") -> list[TranslatedDubSegment]:
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

        self.manifest.record_step(
            step_name="S5",
            config_hash=s5_hash,
            outputs={"translated": str(translated_json.relative_to(self.project_dir))},
            metadata={"elapsed_s": round(elapsed, 2), "total_translated": len(translated_segs)},
        )
        self._progress("S5", 1.0)
        return translated_segs

    def _get_voice_backend(self) -> VoiceBackend:
        b_name = (self.config.tts.backend or "").lower()
        if self.config.tts.mock or b_name == "mock":
            return MockVoiceBackend()
        elif b_name in ("openvoice", "clone", "voice_clone"):
            return OpenVoiceBackend(
                device=self.config.demucs.device,
                pitch_match_mode=self.config.tts.pitch_match_mode,
                pitch_shift_semitones=self.config.tts.pitch_shift_semitones,
            )
        elif b_name == "gtts":
            return GTTSVoiceBackend(lang="vi")
        elif b_name == "f5":
            return F5VoiceBackend(device=self.config.demucs.device)
        # Default to OpenVoice for real voice cloning if reference audio is available
        return OpenVoiceBackend(
            device=self.config.demucs.device,
            pitch_match_mode=self.config.tts.pitch_match_mode,
            pitch_shift_semitones=self.config.tts.pitch_shift_semitones,
        )

    def _sync_manifest_with_config(self, up_to_step: str) -> None:
        """Invalidate downstream cached steps if an earlier step's config hash changed."""
        order = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9"]
        if up_to_step not in order:
            return
        limit = order.index(up_to_step)
        for step in order[: limit + 1]:
            rec = self.manifest.steps.get(step)
            if rec is not None and rec.config_hash != self.config.get_step_hash(step):
                self.manifest.invalidate_from(step)
                return

    def run_s6_tts(self) -> list[dict]:
        """S6: Synthesize Vietnamese voice for each translated segment."""
        self._progress("S6", 0.0)
        self._sync_manifest_with_config("S6")
        s6_hash = self.config.get_step_hash("S6")

        translated_segs = self.run_s5_translation()
        profiles = self.run_s4_voice_profiles()

        # Check existing wav files
        tts_results = []
        all_exist = True
        for s in translated_segs:
            wav_file = self.tts_dir / f"seg_{s.id:04d}.wav"
            if not wav_file.exists() or wav_file.stat().st_size == 0:
                all_exist = False
                break

        if all_exist and self.manifest.is_step_valid("S6", s6_hash):
            self._progress("S6", 1.0)
            for s in translated_segs:
                wav_file = self.tts_dir / f"seg_{s.id:04d}.wav"
                tts_results.append({
                    "id": s.id,
                    "start": s.start,
                    "end": s.end,
                    "duration": s.duration,
                    "speaker_id": s.speaker_id,
                    "raw_wav": wav_file,
                })
            return tts_results

        # S6 cache missed -> invalidate downstream S7-S9 so timing/mixing/muxing rebuild
        self.manifest.invalidate_from("S7")

        speaker_voice_map = self.config.tts.speaker_voice_map or {}
        has_external_map = any(
            v and v not in (SPECIAL_SELF, SPECIAL_ORIGINAL)
            for v in speaker_voice_map.values()
        )
        voice_bank = VoiceBankManager(self.config.tts.voice_bank_dir) if has_external_map else None

        orig_dialogue_audio: Optional[np.ndarray] = None
        dialogue_wav_path = self.audio_dir / "dialogue.wav"

        backend = self._get_voice_backend()
        t0 = time.time()
        try:
            for idx, s in enumerate(translated_segs):
                if self._cancelled:
                    break
                wav_file = self.tts_dir / f"seg_{s.id:04d}.wav"
                raw_mapped = speaker_voice_map.get(s.speaker_id, SPECIAL_SELF)
                mapped_spec = VoiceBankManager._normalize_choice_id(raw_mapped) or SPECIAL_SELF

                if mapped_spec == SPECIAL_ORIGINAL:
                    if orig_dialogue_audio is None and dialogue_wav_path.exists():
                        orig_dialogue_audio = load_wav_float32(dialogue_wav_path, target_sr=24000)
                    sr = 24000
                    if orig_dialogue_audio is not None and len(orig_dialogue_audio) > 0:
                        i0 = max(0, int(s.start * sr))
                        i1 = max(i0 + 1, min(len(orig_dialogue_audio), int(s.end * sr)))
                        audio = orig_dialogue_audio[i0:i1].astype(np.float32)
                    else:
                        audio = np.zeros(int(max(0.1, s.duration) * sr), dtype=np.float32)
                else:
                    spk_prof = profiles.get(s.speaker_id)
                    ref_path = self.project_dir / spk_prof.reference_wav if spk_prof else None
                    ref_text = getattr(spk_prof, "reference_text", None) or s.original_text
                    effective_speaker_key = s.speaker_id

                    if mapped_spec and mapped_spec != SPECIAL_SELF and voice_bank is not None:
                        resolved_ref = voice_bank.resolve_voice(mapped_spec)
                        if resolved_ref is not None and resolved_ref.exists():
                            ref_path = resolved_ref
                            effective_speaker_key = f"{s.speaker_id}::{mapped_spec}"
                            meta_json = resolved_ref.with_name("clone_meta.json")
                            if meta_json.exists():
                                try:
                                    meta = json.loads(meta_json.read_text(encoding="utf-8"))
                                    if meta.get("reference_text"):
                                        ref_text = str(meta["reference_text"])
                                except Exception:
                                    pass

                    audio, sr = backend.synthesize(
                        text=s.translated_text,
                        ref_audio_path=ref_path,
                        speed=self.config.tts.default_speed,
                        target_duration=s.duration,
                        speaker_id=effective_speaker_key,
                        ref_text=ref_text,
                    )

                _write_wav_pcm16(wav_file, audio, sr)
                tts_results.append({
                    "id": s.id,
                    "start": s.start,
                    "end": s.end,
                    "duration": s.duration,
                    "speaker_id": s.speaker_id,
                    "raw_wav": wav_file,
                    "sr": sr,
                })
                self._progress("S6", (idx + 1) / len(translated_segs))
        finally:
            if hasattr(backend, "_converter") and getattr(backend, "_converter") is not None:
                try:
                    setattr(backend, "_converter", None)
                    import gc
                    import torch
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

        elapsed = time.time() - t0
        self.manifest.record_step(
            step_name="S6",
            config_hash=s6_hash,
            outputs={"tts_dir": str(self.tts_dir.relative_to(self.project_dir))},
            metadata={"elapsed_s": round(elapsed, 2), "total_lines": len(tts_results)},
        )
        self._progress("S6", 1.0)
        return tts_results

    def run_s7_snapping(self) -> list[dict]:
        """S7: Align durations to target video timing via speedup or silence padding."""
        self._progress("S7", 0.0)
        self._sync_manifest_with_config("S7")
        s7_hash = self.config.get_step_hash("S7")

        tts_items = self.run_s6_tts()
        synced_clips = []

        all_exist = True
        for it in tts_items:
            sync_wav = self.tts_dir / f"seg_{it['id']:04d}_synced.wav"
            if not sync_wav.exists():
                all_exist = False
                break

        if all_exist and self.manifest.is_step_valid("S7", s7_hash):
            self._progress("S7", 1.0)
            for it in tts_items:
                sync_wav = self.tts_dir / f"seg_{it['id']:04d}_synced.wav"
                with wave.open(str(sync_wav), "rb") as wf:
                    raw = wf.readframes(wf.getnframes())
                    sr = wf.getframerate()
                    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                synced_clips.append({
                    "id": it["id"],
                    "start": it["start"],
                    "end": it["end"],
                    "audio": audio,
                    "sr": sr,
                })
            return synced_clips

        self.manifest.invalidate_from("S8")
        t0 = time.time()
        for idx, it in enumerate(tts_items):
            raw_wav = it["raw_wav"]
            sync_wav = self.tts_dir / f"seg_{it['id']:04d}_synced.wav"

            with wave.open(str(raw_wav), "rb") as wf:
                raw = wf.readframes(wf.getnframes())
                sr = wf.getframerate()
                audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

            aligned_audio, align_info = align_audio_to_target_duration(
                audio,
                sr=sr,
                target_duration=it["duration"],
                min_speed=self.config.tts.min_speed,
                max_speedup=self.config.tts.max_speedup,
            )
            _write_wav_pcm16(sync_wav, aligned_audio, sr)
            synced_clips.append({
                "id": it["id"],
                "start": it["start"],
                "end": it["end"],
                "audio": aligned_audio,
                "sr": sr,
            })
            self._progress("S7", (idx + 1) / len(tts_items))

        elapsed = time.time() - t0
        self.manifest.record_step(
            step_name="S7",
            config_hash=s7_hash,
            outputs={"synced_dir": str(self.tts_dir.relative_to(self.project_dir))},
            metadata={"elapsed_s": round(elapsed, 2)},
        )
        self._progress("S7", 1.0)
        return synced_clips

    def run_s8_mixing(self) -> Path:
        """S8: Mix voice clips with background music using ducking."""
        self._progress("S8", 0.0)
        self._sync_manifest_with_config("S8")
        s8_hash = self.config.get_step_hash("S8")
        mix_out = self.mix_dir / "dub_mix.wav"

        if self.manifest.is_step_valid("S8", s8_hash, [mix_out]):
            self._progress("S8", 1.0)
            return mix_out

        synced_clips = self.run_s7_snapping()
        bgm_wav = self.audio_dir / "bgm.wav"

        self.manifest.invalidate_from("S9")
        t0 = time.time()
        mix_audio_tracks(
            bgm_path=bgm_wav,
            vocal_clips=synced_clips,
            output_path=mix_out,
            ducking_db=self.config.mixing.ducking_db,
            target_sr=self.config.mixing.target_sr,
        )
        elapsed = time.time() - t0

        self.manifest.record_step(
            step_name="S8",
            config_hash=s8_hash,
            outputs={"dub_mix": str(mix_out.relative_to(self.project_dir))},
            metadata={"elapsed_s": round(elapsed, 2)},
        )
        self._progress("S8", 1.0)
        return mix_out

    def run_s9_muxing(self) -> tuple[Path, Path]:
        """S9: Mux dubbed audio into video (-c:v copy) and export Vietnamese SRT."""
        self._progress("S9", 0.0)
        self._sync_manifest_with_config("S9")
        s9_hash = self.config.get_step_hash("S9")

        stem = self.video_path.stem
        final_video = self.out_dir / f"{stem}_dub_vi.mp4"
        final_srt = self.out_dir / f"{stem}_dub_vi.srt"

        if self.manifest.is_step_valid("S9", s9_hash, [final_video, final_srt]):
            self._progress("S9", 1.0)
            return final_video, final_srt

        dub_mix_wav = self.run_s8_mixing()
        translated_segs = self.run_s5_translation()

        t0 = time.time()
        # 1. Export SRT
        export_srt_file(translated_segs, final_srt)
        # 2. Mux video with audio
        mux_video_with_audio(self.video_path, dub_mix_wav, final_video)
        elapsed = time.time() - t0

        self.manifest.record_step(
            step_name="S9",
            config_hash=s9_hash,
            outputs={
                "video": str(final_video.relative_to(self.project_dir)),
                "srt": str(final_srt.relative_to(self.project_dir)),
            },
            metadata={"elapsed_s": round(elapsed, 2)},
        )
        self._progress("S9", 1.0)
        return final_video, final_srt

    def run_full_pipeline(self, context_notes: str = "") -> dict:
        """Execute full 1-click Dubbing Pipeline from S0 to S9."""
        t_start = time.time()
        if (context_notes or "").strip():
            self.run_s5_translation(context_notes=context_notes)
        vid, srt = self.run_s9_muxing()
        total_time = time.time() - t_start

        return {
            "status": "completed",
            "video_path": str(vid),
            "srt_path": str(srt),
            "project_dir": str(self.project_dir),
            "total_elapsed_s": round(total_time, 2),
        }
