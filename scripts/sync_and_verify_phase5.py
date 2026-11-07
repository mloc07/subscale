"""Sync Phase 5 code to Colab and execute full MVP pipeline run."""

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.remote_colab import upload_file, run_remote

files = [
    "src/ecoup/dubbing/__init__.py",
    "src/ecoup/dubbing/config.py",
    "src/ecoup/dubbing/manifest.py",
    "src/ecoup/dubbing/audio.py",
    "src/ecoup/dubbing/asr.py",
    "src/ecoup/dubbing/diarization.py",
    "src/ecoup/dubbing/voice_profile.py",
    "src/ecoup/dubbing/translate.py",
    "src/ecoup/dubbing/tts.py",
    "src/ecoup/dubbing/snapping.py",
    "src/ecoup/dubbing/mixing.py",
    "src/ecoup/dubbing/mux.py",
    "src/ecoup/dubbing/pipeline.py",
    "scripts/verify_phase5_colab.py",
]

def main():
    print("Uploading Phase 5 files to Colab...")
    for f in files:
        target = f"/content/upscale_video/{f}" if f.startswith("src") else f"/content/{Path(f).name}"
        res = upload_file(f, target)
        print(f"  {f} -> {target}: {res.get('status')}")

    print("\nRunning Phase 5 full end-to-end pipeline on Colab GPU...")
    res = run_remote("python3 /content/verify_phase5_colab.py", timeout=600)
    print("Verification exit code:", res.get("exit_code"))
    print("Output:\n", res.get("output", ""))

if __name__ == "__main__":
    main()
