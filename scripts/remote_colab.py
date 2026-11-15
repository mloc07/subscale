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

DEFAULT_BRIDGE_URL = "https://concluded-boot-bandwidth-societies.trycloudflare.com/"


def get_bridge_url() -> str:
    return os.environ.get("COLAB_BRIDGE_URL", DEFAULT_BRIDGE_URL)


def run_remote(cmd: str, cwd: str = "/content", timeout: int = 300) -> dict:
    url = get_bridge_url().rstrip("/") + "/run"
    payload = json.dumps({"cmd": cmd, "cwd": cwd}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json", "User-Agent": "Antigravity/ColabBridge"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def upload_file(local_path: str | Path, remote_path: str, chunk_size: int = 3 * 1024 * 1024) -> dict:
    local_path = Path(local_path)
    raw = local_path.read_bytes()
    url = get_bridge_url().rstrip("/") + "/write"
    if not raw:
        payload = json.dumps({"path": remote_path, "data_b64": "", "append": False}).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json", "User-Agent": "Antigravity/ColabBridge"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode())

    total = 0
    for idx in range(0, len(raw), chunk_size):
        chunk = raw[idx : idx + chunk_size]
        data_b64 = base64.b64encode(chunk).decode("ascii")
        payload = json.dumps({"path": remote_path, "data_b64": data_b64, "append": idx > 0}).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json", "User-Agent": "Antigravity/ColabBridge"})
        with urllib.request.urlopen(req, timeout=180) as resp:
            res = json.loads(resp.read().decode())
        total += int(res.get("size", len(chunk)))
    return {"status": "ok", "size": total}


def download_file(remote_path: str, local_path: str | Path, chunk_size: int = 3 * 1024 * 1024) -> Path:
    """Download a file from Colab in safe base64 chunks."""
    out_p = Path(local_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    sz_res = run_remote(f"python3 -c \"import pathlib; print(pathlib.Path({remote_path!r}).stat().st_size)\"")
    if sz_res.get("exit_code") != 0:
        raise RuntimeError(f"Remote file not found {remote_path}: {sz_res.get('output')}")
    total_size = int(sz_res["output"].strip().splitlines()[-1])

    chunks: list[bytes] = []
    offset = 0
    while offset < total_size or total_size == 0:
        cmd = (
            "python3 -c \""
            f"import base64; f=open({remote_path!r},'rb'); f.seek({offset}); "
            f"d=f.read({chunk_size}); f.close(); print(base64.b64encode(d).decode('ascii'))\""
        )
        res = run_remote(cmd, timeout=180)
        if res.get("exit_code") != 0:
            raise RuntimeError(f"Failed downloading chunk at {offset} from {remote_path}: {res.get('output')}")
        b64_str = "".join(res.get("output", "").split())
        data = base64.b64decode(b64_str)
        chunks.append(data)
        offset += len(data)
        if total_size == 0 or not data:
            break
    out_p.write_bytes(b"".join(chunks))
    return out_p

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
