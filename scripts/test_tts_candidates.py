"""Benchmark multiple TTS candidates on Colab."""

import os
import subprocess
import time

def test_tts():
    print("=== Testing TTS Candidates ===")
    
    # Candidate 1: Edge-TTS
    print("\n--- 1. Testing Edge-TTS ---")
    try:
        import edge_tts
        import asyncio
        
        async def run_edge():
            # Test HoaiMyNeural and NamMinhNeural
            for voice in ["vi-VN-HoaiMyNeural", "vi-VN-NamMinhNeural"]:
                try:
                    out = f"/content/phase0_test/edge_{voice}.mp3"
                    comm = edge_tts.Communicate("Làm sao biết được mình có gì nếu không thử sức?", voice=voice)
                    await comm.save(out)
                    print(f"Edge-TTS ({voice}) SUCCESS: {os.path.exists(out)} ({os.path.getsize(out)} bytes)")
                except Exception as e:
                    print(f"Edge-TTS ({voice}) ERROR: {e}")
                    
        asyncio.run(run_edge())
    except Exception as e:
        print("Edge-TTS failed:", e)

    # Candidate 2: gTTS (Google Translate TTS)
    print("\n--- 2. Testing gTTS ---")
    try:
        from gtts import gTTS
        out = "/content/phase0_test/gtts_vi.mp3"
        tts = gTTS("Làm sao biết được mình có gì nếu không thử sức?", lang="vi")
        tts.save(out)
        print(f"gTTS SUCCESS: {os.path.exists(out)} ({os.path.getsize(out)} bytes)")
    except Exception as e:
        print("gTTS failed:", e)

    # Candidate 3: OpenVoice v2 (Tone Color Converter)
    print("\n--- 3. Checking OpenVoice v2 availability ---")
    try:
        p = subprocess.run(["pip", "install", "-q", "git+https://github.com/myshell-ai/OpenVoice.git"], capture_output=True, text=True)
        print(f"OpenVoice install exit code: {p.returncode}")
        if p.returncode != 0:
            print("OpenVoice install stderr:", p.stderr[:300])
    except Exception as e:
        print("OpenVoice check failed:", e)

if __name__ == "__main__":
    test_tts()
