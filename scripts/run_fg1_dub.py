"""Launch and monitor full voice clone dubbing on FamilyGuy1_test_dub.mp4."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.remote_colab import write_remote_text, run_remote

def main():
    print("1. Preparing launch command for FamilyGuy1 voice cloning...")
    # Clean previous run if exists
    run_remote("rm -rf /content/tests_dub/dub_project_FamilyGuy1_test_dub /content/dub_fg1.log")

    cmd = (
        "nohup python3 -m ecoup.cli dub "
        "-i /content/tests_dub/FamilyGuy1_test_dub.mp4 "
        "--backend openvoice "
        "--whisper-model base "
        "--translate-model gemini-2.5-flash "
        "> /content/dub_fg1.log 2>&1 &"
    )
    print("Launching pipeline...")
    res = run_remote(cmd, cwd="/content/upscale_video")
    print("Launch response:", res)

    print("2. Monitoring pipeline progress...")
    t0 = time.time()
    last_log = ""
    while time.time() - t0 < 600:
        time.sleep(5)
        # Check if process is still running
        ps_res = run_remote("pgrep -f 'ecoup.cli dub'")
        pids = ps_res.get("output", "").strip()

        # Read log tail
        log_res = run_remote("tail -n 15 /content/dub_fg1.log")
        cur_log = log_res.get("output", "").strip()
        if cur_log != last_log:
            print(f"[{int(time.time() - t0)}s]\n{cur_log}\n---")
            last_log = cur_log

        if not pids:
            print("Pipeline process completed!")
            break

    # Read full manifest
    print("\n3. Reading project manifest...")
    man_res = run_remote("cat /content/tests_dub/dub_project_FamilyGuy1_test_dub/manifest.json")
    try:
        man = json.loads(man_res.get("output", "{}"))
        print(json.dumps(man, ensure_ascii=False, indent=2))
    except Exception:
        print("Manifest output:", man_res.get("output"))

    # Check output video
    out_res = run_remote("ls -la /content/tests_dub/dub_project_FamilyGuy1_test_dub/out/")
    print("\nOutput directory:", out_res.get("output"))

if __name__ == "__main__":
    main()
