"""Project Manifest and step cache manager."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


@dataclass
class StepRecord:
    status: str  # "completed", "failed", "pending"
    config_hash: str
    timestamp: float = field(default_factory=time.time)
    outputs: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class ProjectManifest:
    def __init__(self, project_dir: str | Path, video_path: str = ""):
        self.project_dir = Path(project_dir)
        self.manifest_path = self.project_dir / "manifest.json"
        self.video_path = str(video_path)
        self.version = "1.0"
        self.created_at = time.time()
        self.updated_at = time.time()
        self.steps: dict[str, StepRecord] = {}

    @classmethod
    def load_or_create(cls, project_dir: str | Path, video_path: str = "") -> ProjectManifest:
        project_dir = Path(project_dir)
        project_dir.mkdir(parents=True, exist_ok=True)
        manifest_file = project_dir / "manifest.json"

        instance = cls(project_dir, video_path)
        if manifest_file.exists():
            try:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                instance.version = data.get("version", "1.0")
                instance.created_at = data.get("created_at", time.time())
                instance.updated_at = data.get("updated_at", time.time())
                if not instance.video_path and "video_path" in data:
                    instance.video_path = data["video_path"]
                for step_name, s_data in data.get("steps", {}).items():
                    instance.steps[step_name] = StepRecord(
                        status=s_data.get("status", "pending"),
                        config_hash=s_data.get("config_hash", ""),
                        timestamp=s_data.get("timestamp", 0.0),
                        outputs=s_data.get("outputs", {}),
                        metadata=s_data.get("metadata", {}),
                    )
            except Exception:
                # If corrupt, re-initialize cleanly
                instance.save()
        else:
            instance.save()
        return instance

    def save(self) -> None:
        self.updated_at = time.time()
        data = {
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "video_path": self.video_path,
            "steps": {k: asdict(v) for k, v in self.steps.items()},
        }
        self.project_dir.mkdir(parents=True, exist_ok=True)
        temp_file = self.manifest_path.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        temp_file.replace(self.manifest_path)

    def is_step_valid(
        self,
        step_name: str,
        config_hash: str,
        expected_outputs: Sequence[str | Path] = (),
    ) -> bool:
        """Check if step was previously completed with matching config and all output files exist."""
        if step_name not in self.steps:
            return False
        record = self.steps[step_name]
        if record.status != "completed":
            return False
        if record.config_hash != config_hash:
            return False
        for out in expected_outputs:
            p = Path(out)
            if not p.is_absolute():
                p = self.project_dir / p
            if not p.exists() or p.stat().st_size == 0:
                return False
        return True

    def record_step(
        self,
        step_name: str,
        config_hash: str,
        outputs: dict[str, str],
        metadata: Optional[dict[str, Any]] = None,
        status: str = "completed",
    ) -> None:
        self.steps[step_name] = StepRecord(
            status=status,
            config_hash=config_hash,
            timestamp=time.time(),
            outputs=outputs,
            metadata=metadata or {},
        )
        self.save()

    def invalidate_from(self, step_name: str) -> None:
        """Invalidate step and all subsequent steps."""
        all_steps = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10"]
        if step_name in all_steps:
            idx = all_steps.index(step_name)
            for s in all_steps[idx:]:
                if s in self.steps:
                    del self.steps[s]
            self.save()
