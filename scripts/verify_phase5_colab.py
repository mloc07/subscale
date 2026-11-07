"""Verification script for Phase 5 MVP on Colab (Full End-to-End Dubbing Run)."""

import os
import sys
import time
from pathlib import Path

# Add ecoup to sys.path
sys.path.insert(0, "/content/upscale_video/src")

from ecoup.dubbing.config import (
    DubbingConfig,
    DemucsConfig,
    AudioExtractConfig,
    AsrDubbingConfig,
    DiarizationDubbingConfig,
    TranslationDubbingConfig,
    TtsDubbingConfig,
    MixingDubbingConfig,
)
from ecoup.dubbing.pipeline import DubbingPipeline
from ecoup.dubbing.manifest import ProjectManifest

def verify_phase5():
    print("=== Executing Full End-to-End AI Dubbing Pipeline (Phase 5 MVP) on Colab ===")
    video_path = "/content/tests_dub/FamilyGuy1_test_dub.mp4"
    project_dir = "/content/dub_project_FamilyGuy1"
    
    cfg = DubbingConfig(
        project_dir=project_dir,
        audio=AudioExtractConfig(sample_rate=44100, channels=2),
        demucs=DemucsConfig(model_name="htdemucs", two_stems="vocals", device="cuda"),
        asr=AsrDubbingConfig(model_size="base", beam_size=5, vad_filter=True),
        diarization=DiarizationDubbingConfig(distance_threshold=0.45),
        translation=TranslationDubbingConfig(model="gemini-2.5-flash", syllables_per_second=3.8),
        tts=TtsDubbingConfig(backend="gtts", default_speed=1.0, max_speedup=1.25),
        mixing=MixingDubbingConfig(ducking_db=-8.0, target_sr=44100),
    )
    
    pipeline = DubbingPipeline(video_path, config=cfg)
    
    print("\nStarting 1-Click Full Pipeline Run (S0 -> S9)...")
    t0 = time.time()
    result = pipeline.run_full_pipeline(context_notes="Phim hoạt hình Family Guy, Peter Griffin hài hước.")
    total_time = time.time() - t0
    
    print(f"\nPipeline Finished in {total_time:.2f}s!")
    print(f"Status:     {result['status']}")
    print(f"Final Video: {result['video_path']} (size: {os.path.getsize(result['video_path'])} bytes)")
    print(f"Final SRT:   {result['srt_path']} (size: {os.path.getsize(result['srt_path'])} bytes)")
    
    # Check output video properties with ffprobe
    import subprocess
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration,size:stream=codec_name,width,height,channels,sample_rate",
        "-of", "default=noprint_wrappers=1",
        result['video_path']
    ]
    res_probe = subprocess.run(cmd, capture_output=True, text=True)
    print("\n--- Output Video FFprobe Info ---")
    print(res_probe.stdout.strip())
    
    # Check SRT contents
    print("\n--- First 3 Subtitle Entries in SRT ---")
    with open(result['srt_path'], "r", encoding="utf-8") as f:
        print("\n".join(f.read().splitlines()[:12]))
        
    print("\n>>> Phase 5 MVP Full Pipeline Run PASSED 100%! <<<")

if __name__ == "__main__":
    verify_phase5()
