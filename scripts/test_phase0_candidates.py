"""Test F5-TTS, Pyannote and SpeechBrain on Colab."""

import sys
import subprocess

def test_pip_packages():
    results = {}
    
    print("--- 1. Testing f5-tts install ---")
    p = subprocess.run(["pip", "install", "-q", "f5-tts"], capture_output=True, text=True)
    results["f5-tts"] = {"status": p.returncode, "err": p.stderr[-400:] if p.returncode != 0 else ""}
    print("F5-TTS result:", p.returncode)

    print("--- 2. Testing pyannote.audio install ---")
    p = subprocess.run(["pip", "install", "-q", "pyannote.audio"], capture_output=True, text=True)
    results["pyannote.audio"] = {"status": p.returncode, "err": p.stderr[-400:] if p.returncode != 0 else ""}
    print("pyannote.audio result:", p.returncode)

    print("--- 3. Testing speechbrain install ---")
    p = subprocess.run(["pip", "install", "-q", "speechbrain"], capture_output=True, text=True)
    results["speechbrain"] = {"status": p.returncode, "err": p.stderr[-400:] if p.returncode != 0 else ""}
    print("speechbrain result:", p.returncode)

    print("\nSummary:", results)

if __name__ == "__main__":
    test_pip_packages()
