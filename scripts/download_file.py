"""Download large binary files from Colab in chunks via bridge /run."""

import base64
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.remote_colab import run_remote

def download_file_chunked(remote_path: str, local_path: str | Path, chunk_size: int = 2 * 1024 * 1024):
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    # Get file size
    size_res = run_remote(f"stat -c %s '{remote_path}'")
    total_size = int(size_res.get("output", "0").strip())
    print(f"Downloading {remote_path} ({round(total_size / (1024*1024), 2)} MB) to {local_path}...")

    offset = 0
    t0 = time.time()
    with open(local_path, "wb") as f_out:
        while offset < total_size:
            read_len = min(chunk_size, total_size - offset)
            py_cmd = (
                f"python3 -c \""
                f"import base64; "
                f"f = open('{remote_path}', 'rb'); "
                f"f.seek({offset}); "
                f"data = f.read({read_len}); "
                f"print(base64.b64encode(data).decode('ascii'))\""
            )
            res = run_remote(py_cmd)
            raw_b64 = res.get("output", "").strip()
            if not raw_b64:
                raise RuntimeError(f"Failed to read chunk at offset {offset}")
            chunk_bytes = base64.b64decode(raw_b64)
            f_out.write(chunk_bytes)
            offset += len(chunk_bytes)
            pct = int((offset / total_size) * 100)
            elapsed = time.time() - t0
            print(f"  -> {pct}% ({round(offset / (1024*1024), 2)} MB, {round(elapsed, 1)}s)")

    print(f"Downloaded completed: {local_path} ({local_path.stat().st_size} bytes in {round(time.time() - t0, 1)}s)")

if __name__ == "__main__":
    remote_p = "/content/tests_dub/dub_project_FamilyGuy2_test_dub/out/FamilyGuy2_test_dub_dub_vi.mp4"
    local_p = "outputs/dubbed/FamilyGuy2_test_dub_vi.mp4"
    download_file_chunked(remote_p, local_p)
