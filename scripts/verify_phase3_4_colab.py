"""Verification script for Phase 3 & 4 on Colab with FamilyGuy1 video."""

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
)
from ecoup.dubbing.pipeline import DubbingPipeline
from ecoup.dubbing.manifest import ProjectManifest

def verify_phase3_4():
    print("=== Verifying Phase 3 (Voice Profile) & Phase 4 (Translation) on Colab ===")
    video_path = "/content/tests_dub/FamilyGuy1_test_dub.mp4"
    project_dir = "/content/dub_project_FamilyGuy1"
    
    cfg = DubbingConfig(
        project_dir=project_dir,
        audio=AudioExtractConfig(sample_rate=44100, channels=2),
        demucs=DemucsConfig(model_name="htdemucs", two_stems="vocals", device="cuda"),
        asr=AsrDubbingConfig(model_size="base", beam_size=5, vad_filter=True),
        diarization=DiarizationDubbingConfig(distance_threshold=0.45),
        translation=TranslationDubbingConfig(model="gemini-2.5-flash", syllables_per_second=3.8)
    )
    
    pipeline = DubbingPipeline(video_path, config=cfg)
    
    print("\n--- 1. Run S4: Voice Profile Extraction ---")
    t0 = time.time()
    profiles = pipeline.run_s4_voice_profiles()
    s4_time = time.time() - t0
    print(f"S4 Completed in {s4_time:.2f}s! Extracted {len(profiles)} voice profiles:")
    for spk, vp in profiles.items():
        print(f"  [{spk} - {vp.character_name}]: Score={vp.quality_score}, IsGood={vp.is_good_reference}, Ref={vp.reference_wav}, Flags={vp.flags}")
        
    print("\n--- 2. Run S5: Gemini Translation with Duration Budget ---")
    t0 = time.time()
    translated_segs = pipeline.run_s5_translation(context_notes="Phim hoạt hình Family Guy, nhân vật chính Peter Griffin hài hước, phóng khoáng.")
    s5_time = time.time() - t0
    print(f"S5 Completed in {s5_time:.2f}s! Translated {len(translated_segs)} segments.")
    
    overflow_count = 0
    print("\nSample translated lines:")
    for item in translated_segs[:8]:
        status = "OK" if not item.overflow_flag else "OVERFLOW"
        if item.overflow_flag:
            overflow_count += 1
        print(f"  [id {item.id}|{item.duration}s|{item.speaker_id}] [{status}] ({item.syllable_count}/{item.max_budget} từ):")
        print(f"      Gốc: \"{item.original_text}\"")
        print(f"      Việt: \"{item.translated_text}\"")
        
    print(f"\nTotal overflowed sentences: {overflow_count}/{len(translated_segs)}")
    
    print("\n--- 3. Verify Cache on Second Run ---")
    t0 = time.time()
    pipeline2 = DubbingPipeline(video_path, config=cfg)
    c_s4 = pipeline2.run_s4_voice_profiles()
    c_s5 = pipeline2.run_s5_translation()
    cache_time = time.time() - t0
    print(f"Cached S4 & S5 executed in: {cache_time:.4f}s (Instant!)")
    
    assert len(c_s4) == len(profiles)
    assert len(c_s5) == len(translated_segs)
    assert cache_time < 0.2
    
    print("\n>>> Phase 3 & 4 Verification PASSED 100%! <<<")

if __name__ == "__main__":
    verify_phase3_4()
