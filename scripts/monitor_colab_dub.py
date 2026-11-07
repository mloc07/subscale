"""Launch dubbing pipeline in background on Colab and monitor progress via manifest/log."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.remote_colab import run_remote

def main():
    # 1. Clean previous partial run on Colab if any
    print("Resetting previous project artifacts on Colab...")
    run_remote("rm -rf /content/tests_dub/dub_project_FamilyGuy2_test_dub/voices /content/tests_dub/dub_project_FamilyGuy2_test_dub/translation /content/tests_dub/dub_project_FamilyGuy2_test_dub/tts /content/tests_dub/dub_project_FamilyGuy2_test_dub/mix /content/tests_dub/dub_project_FamilyGuy2_test_dub/out")

    # Invalidate manifest from S3 so it re-runs S3 with clamped speakers and S4-S9
    py_reset = (
        "python3 -c \""
        "import json, pathlib; "
        "p = pathlib.Path('/content/tests_dub/dub_project_FamilyGuy2_test_dub/manifest.json'); "
        "d = json.load(open(p)); "
        "d['steps'] = {k: v for k, v in d.get('steps', {}).items() if k in ('S0', 'S1', 'S2')}; "
        "json.dump(d, open(p, 'w'), indent=2); "
        "print('Manifest reset to S2 done')\""
    )
    run_remote(py_reset)

    # 2. Launch background job
    cmd = (
        "nohup python3 -m ecoup.cli dub "
        "-i /content/tests_dub/FamilyGuy2_test_dub.mp4 "
        "--backend gtts "
        "--whisper-model large-v3 "
        "--translate-model gemini-2.5-flash "
        "--context 'Phim hoat hinh Family Guy, hai huoc, xưng ho Peter/Lois/Brian/Stewie' "
        "> /content/dub_familyguy2.log 2>&1 & echo $!"
    )
    print("Launching background dubbing job on Colab...")
    res = run_remote(cmd)
    pid = res.get("output", "").strip()
    print(f"Launched job with PID: {pid}")

    # 3. Monitor progress
    manifest_p = "/content/tests_dub/dub_project_FamilyGuy2_test_dub/manifest.json"
    t_start = time.time()
    last_log_pos = 0

    while True:
        time.sleep(4)
        elapsed = round(time.time() - t_start, 1)

        # Check if process is still running
        ps_res = run_remote(f"ps -p {pid} -o pid=")
        running = bool(ps_res.get("output", "").strip())

        # Read last lines of log
        log_res = run_remote("tail -n 6 /content/dub_familyguy2.log")
        tail_log = log_res.get("output", "").strip()

        # Check manifest
        mf_res = run_remote(f"cat {manifest_p}")
        completed_steps = []
        try:
            mf = json.loads(mf_res.get("output", "{}"))
            for s, sdata in mf.get("steps", {}).items():
                if sdata.get("status") == "completed":
                    completed_steps.append(s)
        except Exception:
            pass

        print(f"[{elapsed}s] Steps completed: {completed_steps} | Process running: {running}")
        if tail_log:
            print(f"  Log tail: {tail_log.splitlines()[-1] if tail_log.splitlines() else ''}")

        if not running:
            print(f"\nProcess {pid} ended after {elapsed}s.")
            break

    # 4. Final verification
    print("\n--- FINAL LOG OUTPUT ---")
    full_log = run_remote("cat /content/dub_familyguy2.log")
    print(full_log.get("output", ""))

    print("\n--- MANIFEST ---")
    final_mf = run_remote(f"cat {manifest_p}")
    print(final_mf.get("output", ""))

if __name__ == "__main__":
    main()
