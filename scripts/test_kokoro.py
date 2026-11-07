"""Test Kokoro-Vietnamese on Colab."""

import subprocess
import os
import time

def test_kokoro_vi():
    print("=== Testing Kokoro-Vietnamese on Colab ===")
    
    # 1. Install dependencies
    print("Installing kokoro or kokoro-onnx / kokoro-vietnamese...")
    p = subprocess.run(["pip", "install", "-q", "kokoro", "misaki[vi]", "soundfile"], capture_output=True, text=True)
    print("Pip install status:", p.returncode)
    
    # Check if kokoro is available
    try:
        from kokoro import KPipeline
        print("KPipeline imported from kokoro!")
        # Let's inspect available languages
        pipeline = KPipeline(lang_code='v') # 'v' for Vietnamese
        print("Vietnamese pipeline created!")
        
        text = "Làm sao biết được mình có gì nếu không thử sức? Đừng chống đẩy một cái, hãy trăm cái luôn."
        t0 = time.time()
        generator = pipeline(text, voice='vf_vietnam', speed=1)
        import soundfile as sf
        all_audio = []
        for i, (gs, ps, audio) in enumerate(generator):
            all_audio.append(audio)
            print(f"Segment {i}: len={len(audio)}")
            
        import numpy as np
        final_audio = np.concatenate(all_audio) if all_audio else None
        t_gen = time.time() - t0
        if final_audio is not None:
            out_file = "/content/phase0_test/kokoro_sample.wav"
            sf.write(out_file, final_audio, 24000)
            print(f"Kokoro generated successfully in {t_gen:.2f}s! File: {out_file}, duration: {len(final_audio)/24000:.2f}s")
            
            # Evaluate with ASR
            from test_phase0_audio import preload_cuda12_libs, load_wav_float32
            preload_cuda12_libs()
            audio_arr = load_wav_float32(out_file)
            from faster_whisper import WhisperModel
            asr = WhisperModel("base", device="cuda", compute_type="float16")
            segs, info = asr.transcribe(audio_arr, language="vi")
            rec = " ".join([s.text.strip() for s in segs])
            print(f"ASR of Kokoro output: '{rec}'")
    except Exception as e:
        import traceback
        print("Kokoro test error:", e)
        traceback.print_exc()

if __name__ == "__main__":
    test_kokoro_vi()
