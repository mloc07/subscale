"""Verification script for Phase 2 on Colab with FamilyGuy1 video."""

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
    DiarizationDubbingConfig
)
from ecoup.dubbing.pipeline import DubbingPipeline
from ecoup.dubbing.manifest import ProjectManifest

def verify_phase2():
    print("=== Verifying Phase 2 (S2: ASR & S3: Diarization) on Colab ===")
    video_path = "/content/tests_dub/FamilyGuy1_test_dub.mp4"
    project_dir = "/content/dub_project_FamilyGuy1"
    
    cfg = DubbingConfig(
        project_dir=project_dir,
        audio=AudioExtractConfig(sample_rate=44100, channels=2),
        demucs=DemucsConfig(model_name="htdemucs", two_stems="vocals", device="cuda"),
        asr=AsrDubbingConfig(model_size="base", beam_size=5, vad_filter=True),
        diarization=DiarizationDubbingConfig(distance_threshold=0.45)
    )
    
    pipeline = DubbingPipeline(video_path, config=cfg)
    
    print("\n--- 1. Run S2: ASR Speech Recognition ---")
    t0 = time.time()
    segments = pipeline.run_s2_asr()
    s2_time = time.time() - t0
    print(f"S2 Completed in {s2_time:.2f}s! Segments count: {len(segments)}")
    for s in segments[:5]:
        print(f"  [{s.start}s -> {s.end}s ({s.duration}s)]: {s.text}")
        
    print("\n--- 2. Run S3: Diarization & Speaker Clustering ---")
    t0 = time.time()
    diarized_segs = pipeline.run_s3_diarization()
    s3_time = time.time() - t0
    print(f"S3 Completed in {s3_time:.2f}s!")
    
    unique_speakers = sorted(list(set(s.speaker_id for s in diarized_segs if s.speaker_id)))
    print(f"Unique speakers identified: {unique_speakers}")
    for spk in unique_speakers:
        spk_segs = [s for s in diarized_segs if s.speaker_id == spk]
        total_time = sum(s.duration for s in spk_segs)
        print(f"  {spk}: {len(spk_segs)} segments, {total_time:.2f}s total dialogue")
        
    preview_path = Path(project_dir) / "voices" / "spk_0" / "preview.wav"
    profile_path = Path(project_dir) / "voices" / "spk_0" / "profile.json"
    print(f"spk_0 preview exists: {preview_path.exists()} ({os.path.getsize(preview_path)} bytes)")
    print(f"spk_0 profile exists: {profile_path.exists()} ({os.path.getsize(profile_path)} bytes)")
    
    print("\n--- 3. Verify Cache on Second Run ---")
    t0 = time.time()
    pipeline2 = DubbingPipeline(video_path, config=cfg)
    c_s2 = pipeline2.run_s2_asr()
    c_s3 = pipeline2.run_s3_diarization()
    cache_time = time.time() - t0
    print(f"Cached S2 & S3 executed in: {cache_time:.4f}s (Instant!)")
    
    assert len(c_s2) == len(segments)
    assert len(c_s3) == len(diarized_segs)
    assert preview_path.exists()
    assert cache_time < 0.2
    
    print("\n>>> Phase 2 Verification PASSED 100%! <<<")

if __name__ == "__main__":
    verify_phase2()
