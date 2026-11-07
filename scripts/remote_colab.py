"""Helper tool to run commands and transfer files to Colab bridge."""

import base64
import json
import os
import sys
from pathlib import Path
import urllib.request

# Avoid UnicodeEncodeError on Windows cp1252 consoles
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

DEFAULT_BRIDGE_URL = "https://subsection-old-none-bidding.trycloudflare.com/"

def get_bridge_url() -> str:
    return os.environ.get("COLAB_BRIDGE_URL", DEFAULT_BRIDGE_URL)

def run_remote(cmd: str, cwd: str = "/content", timeout: int = 300) -> dict:
    url = get_bridge_url().rstrip("/") + "/run"
    payload = json.dumps({"cmd": cmd, "cwd": cwd}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json", "User-Agent": "Antigravity/ColabBridge"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())

def upload_file(local_path: str | Path, remote_path: str) -> dict:
    local_path = Path(local_path)
    url = get_bridge_url().rstrip("/") + "/write"
    data_b64 = base64.b64encode(local_path.read_bytes()).decode("ascii")
    payload = json.dumps({"path": remote_path, "data_b64": data_b64, "append": False}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json", "User-Agent": "Antigravity/ColabBridge"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())

def write_remote_text(text: str, remote_path: str) -> dict:
    url = get_bridge_url().rstrip("/") + "/write"
    data_b64 = base64.b64encode(text.encode("utf-8")).decode("ascii")
    payload = json.dumps({"path": remote_path, "data_b64": data_b64, "append": False}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json", "User-Agent": "Antigravity/ColabBridge"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())

if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = " ".join(sys.argv[1:])
        res = run_remote(cmd)
        print(f"Exit code: {res.get('exit_code')}")
        print(res.get("output", ""))
    else:
        print("Usage: python scripts/remote_colab.py <command>")
