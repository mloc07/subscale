"""Download generated dubbed video and srt from Colab to local machine."""

import base64
import json
import os
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.remote_colab import get_bridge_url

def download_file_from_colab(remote_path: str, local_path: Path):
    local_path.parent.mkdir(parents=True, exist_ok=True)
    # Read file content from Colab via python one-liner running base64
    url = get_bridge_url().rstrip("/") + "/run"
    cmd = f"python3 -c \"import base64; print(base64.b64encode(open('{remote_path}', 'rb').read()).decode('ascii'))\""
    payload = json.dumps({"cmd": cmd, "cwd": "/content"}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json", "User-Agent": "Antigravity/ColabBridge"})
    
    print(f"Downloading {remote_path} -> {local_path}...")
    with urllib.request.urlopen(req, timeout=120) as resp:
        res = json.loads(resp.read().decode())
        raw_b64 = res.get("output", "").strip()
        data = base64.b64decode(raw_b64)
        local_path.write_bytes(data)
        print(f"  Saved {local_path} ({len(data)} bytes)")

def main():
    out_dir = REPO_ROOT / "outputs" / "dubbed"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    remote_video = "/content/dub_project_FamilyGuy1/out/FamilyGuy1_test_dub_dub_vi.mp4"
    remote_srt = "/content/dub_project_FamilyGuy1/out/FamilyGuy1_test_dub_dub_vi.srt"
    
    local_video = out_dir / "FamilyGuy1_test_dub_vi.mp4"
    local_srt = out_dir / "FamilyGuy1_test_dub_vi.srt"
    
    download_file_from_colab(remote_video, local_video)
    download_file_from_colab(remote_srt, local_srt)
    print("\nDownload complete! Files available locally in outputs/dubbed/")

if __name__ == "__main__":
    main()
