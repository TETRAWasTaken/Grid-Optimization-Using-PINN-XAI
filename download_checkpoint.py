#!/usr/bin/env python3
"""
Download trained model checkpoints from the remote Ray GPU host directly
to your local ./outputs/ directory without requiring SSH passwords.
"""

import os
import sys
import base64
import subprocess

def download_latest_checkpoint(host_ip="10.24.14.90", dashboard_port="8265"):
    server_url = f"http://{host_ip}:{dashboard_port}"
    print(f"Connecting to Ray cluster at {server_url} to fetch latest checkpoint...")

    # Remote command to find the most recent checkpoint and encode it to base64
    remote_code = (
        "import glob, os, base64; "
        "paths = glob.glob('/tmp/ray/**/outputs/*checkpoint*.pt', recursive=True); "
        "latest = max(paths, key=os.path.getmtime); "
        "data = open(latest, 'rb').read(); "
        "print('CKPT_DATA:' + base64.b64encode(data).decode('ascii'))"
    )

    import shutil
    ray_bin = shutil.which("ray") or os.path.abspath(".venv/bin/ray")

    cmd = [
        ray_bin, "job", "submit",
        f"--address={server_url}",
        "--",
        "python", "-c", remote_code
    ]

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error communicating with Ray cluster: {e.stderr}")
        sys.exit(1)

    ckpt_b64 = None
    for line in res.stdout.splitlines():
        if line.startswith("CKPT_DATA:"):
            ckpt_b64 = line.split("CKPT_DATA:")[1].strip()
            break

    if not ckpt_b64:
        print("Could not retrieve checkpoint data from host. Full output:")
        print(res.stdout)
        sys.exit(1)

    raw_bytes = base64.b64decode(ckpt_b64)
    os.makedirs("outputs", exist_ok=True)
    target_path = "outputs/best_model_checkpoint.pt"
    with open(target_path, "wb") as f:
        f.write(raw_bytes)
    with open("outputs/model_checkpoint.pt", "wb") as f:
        f.write(raw_bytes)

    print(f"Successfully downloaded checkpoint ({len(raw_bytes)} bytes) to outputs/!")

if __name__ == "__main__":
    ip = sys.argv[1] if len(sys.argv) > 1 else "10.24.14.90"
    download_latest_checkpoint(host_ip=ip)
