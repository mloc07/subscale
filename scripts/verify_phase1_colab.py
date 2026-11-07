"""Verification script for Phase 1 on Colab with FamilyGuy1 video."""

import os
import sys
import time
from pathlib import Path

# Add ecoup to sys.path
sys.path.insert(0, "/content/upscale_video/src")

from ecoup.dubbing.config import DubbingConfig, DemucsConfig, AudioExtractConfig
from ecoup.dubbing.pipeline import DubbingPipeline
from ecoup.dubbing.manifest import ProjectManifest

def verify_phase1():
    print("=== Verifying Phase 1 (S0 & S1) on Colab ===")
    video_path = "/content/tests_dub/FamilyGuy1_test_dub.mp4"
    project_dir = "/content/dub_project_FamilyGuy1"
    
    cfg = DubbingConfig(
        project_dir=project_dir,
        audio=AudioExtractConfig(sample_rate=44100, channels=2),
        demucs=DemucsConfig(model_name="htdemucs", two_stems="vocals", device="cuda")
    )
    
    print("\n--- 1. First Run: Execute S0 and S1 ---")
    t0 = time.time()
    pipeline = DubbingPipeline(video_path, config=cfg)
    
    print("Executing S0 (Audio intake)...")
    orig_wav = pipeline.run_s0_intake()
    print(f"S0 Done: {orig_wav} (size: {os.path.getsize(orig_wav)} bytes)")
    
    print("Executing S1 (Demucs separation)...")
    dialogue_wav, bgm_wav = pipeline.run_s1_separation()
    print(f"S1 Done:")
    print(f"  Dialogue: {dialogue_wav} (size: {os.path.getsize(dialogue_wav)} bytes)")
    print(f"  BGM:      {bgm_wav} (size: {os.path.getsize(bgm_wav)} bytes)")
    print(f"First run total time: {time.time() - t0:.2f}s")
    
    print("\n--- 2. Second Run: Verify Cache / Skip Mechanism ---")
    t0 = time.time()
    pipeline2 = DubbingPipeline(video_path, config=cfg)
    orig_wav2 = pipeline2.run_s0_intake()
    dialogue_wav2, bgm_wav2 = pipeline2.run_s1_separation()
    elapsed2 = time.time() - t0
    print(f"Second run total time: {elapsed2:.4f}s (Instant cache skip!)")
    
    # Check manifest.json contents
    manifest = ProjectManifest.load_or_create(project_dir)
    print("\n--- 3. Manifest Verification ---")
    print(f"Steps recorded: {list(manifest.steps.keys())}")
    for s_name, s_rec in manifest.steps.items():
        print(f"  {s_name}: status={s_rec.status}, hash={s_rec.config_hash[:10]}..., outputs={s_rec.outputs}")

    assert os.path.exists(orig_wav), "original_hq.wav missing"
    assert os.path.exists(dialogue_wav), "dialogue.wav missing"
    assert os.path.exists(bgm_wav), "bgm.wav missing"
    assert elapsed2 < 0.5, f"Cache skip should be instant (<0.5s), got {elapsed2:.4f}s"
    print("\n>>> Phase 1 Verification PASSED 100%! <<<")

if __name__ == "__main__":
    verify_phase1()
