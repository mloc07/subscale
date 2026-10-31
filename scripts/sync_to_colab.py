"""Sync local codebase, API key, and dependencies to remote Google Colab via bridge."""

import base64
import json
import os
from pathlib import Path
import sys
import urllib.request

REPO_ROOT = Path(__file__).resolve().parent.parent

FILES_TO_SYNC = [
    "src/ecoup/subtitle/config.py",
    "src/ecoup/subtitle/audio.py",
    "src/ecoup/subtitle/asr.py",
    "src/ecoup/subtitle/pipeline.py",
    "src/ecoup/subtitle/snapping.py",
    "colab_app.py",
    "scripts/asr_eval.py",
]

def upload(bridge_url: str, local_path: Path, remote_path: str):
    endpoint = bridge_url.rstrip("/") + "/write"
    data_b64 = base64.b64encode(local_path.read_bytes()).decode("ascii")
    payload = json.dumps({"path": remote_path, "data_b64": data_b64, "append": False}).encode("utf-8")
    req = urllib.request.Request(endpoint, data=payload, headers={"Content-Type": "application/json", "User-Agent": "Antigravity/ColabBridge"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())

def run_cmd(bridge_url: str, cmd: str, cwd="/content/upscale_video"):
    endpoint = bridge_url.rstrip("/") + "/run"
    payload = json.dumps({"cmd": cmd, "cwd": cwd}).encode("utf-8")
    req = urllib.request.Request(endpoint, data=payload, headers={"Content-Type": "application/json", "User-Agent": "Antigravity/ColabBridge"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode())

def main():
    bridge_url = sys.argv[1] if len(sys.argv) > 1 else "https://funky-watches-lan-modelling.trycloudflare.com/"
    print(f"Connecting to {bridge_url}...")

    # 1. Sync code files
    for rel in FILES_TO_SYNC:
        loc = REPO_ROOT / rel
        rem = f"/content/upscale_video/{rel}"
        print(f"Uploading {rel} -> {rem}...")
        res = upload(bridge_url, loc, rem)
        print(f"  Result: {res}")

    # 2. Upload Gemini API Key (from env or argument, never hardcoded)
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if len(sys.argv) > 2 and sys.argv[2].strip():
        gemini_key = sys.argv[2].strip()

    if gemini_key:
        endpoint = bridge_url.rstrip("/") + "/write"
        data_b64 = base64.b64encode(gemini_key.encode()).decode("ascii")
        payload = json.dumps({"path": "/content/.gemini_key", "data_b64": data_b64, "append": False}).encode("utf-8")
        req = urllib.request.Request(endpoint, data=payload, headers={"Content-Type": "application/json", "User-Agent": "Antigravity/ColabBridge"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            print("  Gemini Key written to /content/.gemini_key:", resp.read().decode())
    else:
        print("  Notice: No GEMINI_API_KEY provided (skip uploading key).")

    # 3. Check and install packages
    print("Installing jiwer & demucs on Colab...")
    cmd_res = run_cmd(bridge_url, "pip install -q jiwer demucs")
    print("Pip install output:", cmd_res.get("output", "")[:200])

    # 4. Verify git diff on Colab
    diff_res = run_cmd(bridge_url, "git diff --stat")
    print("\nGit Diff Stat on Colab:\n", diff_res.get("output", ""))

if __name__ == "__main__":
    main()
