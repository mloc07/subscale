"""Run dubbing benchmark on FamilyGuy2_test_dub.mp4 via Colab GPU bridge."""

import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.remote_colab import run_remote

def main():
    cmd = (
        "cd /content/upscale_video && "
        "python3 -m ecoup.cli dub -i /content/tests_dub/FamilyGuy2_test_dub.mp4 "
        "--backend gtts "
        "--whisper-model large-v3 "
        "--translate-model gemini-2.5-flash "
        "--context 'Phim hoat hinh Family Guy, boi canh gia dinh va ban be, Peter Griffin xung ho hai huoc'"
    )
    print("Starting multi-speaker dubbing on FamilyGuy2 (4m41s) on Colab L4 GPU...")
    start_t = time.time()
    res = run_remote(cmd, timeout=900)
    elapsed = time.time() - start_t

    print(f"Completed in {elapsed:.1f}s, Exit Code: {res.get('exit_code')}")
    output = res.get("output", "")
    if output:
        print("\n--- OUTPUT ---")
        print(output)

if __name__ == "__main__":
    main()
