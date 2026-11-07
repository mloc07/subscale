"""Sync Phase 1 code to Colab and execute verification."""

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
    "src/ecoup/dubbing/pipeline.py",
    "scripts/verify_phase1_colab.py",
]

def main():
    print("Uploading files to Colab...")
    for f in files:
        target = f"/content/upscale_video/{f}" if f.startswith("src") else f"/content/{Path(f).name}"
        res = upload_file(f, target)
        print(f"  {f} -> {target}: {res.get('status')}")

    print("\nRunning verification script on Colab GPU...")
    res = run_remote("python3 /content/verify_phase1_colab.py", timeout=300)
    print("Verification exit code:", res.get("exit_code"))
    print("Output:\n", res.get("output", ""))

if __name__ == "__main__":
    main()
