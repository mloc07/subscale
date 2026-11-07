"""Dubbing pipeline coordinator."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

from .audio import AudioProcessingError, NoAudioError, extract_original_hq, probe_audio_info, separate_stems
from .config import DubbingConfig
from .manifest import ProjectManifest


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

        # Ensure S0 is completed
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
