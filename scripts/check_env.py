#!/usr/bin/env python3
"""Report the local development environment: OS, CPU, memory, disk, Python, PyTorch device.

Usage: python scripts/check_env.py [--json]
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess  # noqa: S404 - fixed, trusted commands only
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _mem_gb() -> float | None:
    if sys.platform == "darwin":
        out = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "hw.memsize"], capture_output=True, text=True
        )  # noqa: S603
        return round(int(out.stdout.strip()) / 1024**3, 1) if out.returncode == 0 else None
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3, 1)
    except (ValueError, OSError):
        return None


def _tool(name: str) -> str | None:
    path = shutil.which(name)
    if not path:
        return None
    out = subprocess.run([path, "--version"], capture_output=True, text=True)  # noqa: S603
    return (out.stdout or out.stderr).strip().splitlines()[0] if out.returncode == 0 else path


def suggest_batch_size(device: str, mem_gb: float | None) -> int:
    """Starting batch size for 512x512 2-channel U-Net (ResNet-34) training. Tune empirically."""
    if device == "cuda":
        return 16
    if device == "mps":
        return 8 if (mem_gb or 0) >= 16 else 4
    return 2


def main() -> None:
    disk = shutil.disk_usage(REPO)
    info: dict = {
        "os": f"{platform.system()} {platform.mac_ver()[0] or platform.release()}",
        "arch": platform.machine(),
        "cpu_count": os.cpu_count(),
        "memory_gb": _mem_gb(),
        "disk_free_gb": round(disk.free / 1024**3, 1),
        "python": sys.version.split()[0],
        "tools": {t: _tool(t) for t in ["git", "docker", "node", "npm", "psql", "redis-server"]},
    }
    try:
        import torch

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        info["torch"] = {"version": torch.__version__, "device": device}
    except ImportError:
        device = "cpu"
        info["torch"] = None
    info["suggested_batch_size"] = suggest_batch_size(device, info["memory_gb"])

    if "--json" in sys.argv:
        print(json.dumps(info, indent=2))
        return
    for k, v in info.items():
        if isinstance(v, dict):
            print(f"{k}:")
            for kk, vv in v.items():
                print(f"  {kk:<14} {vv if vv is not None else 'not found'}")
        else:
            print(f"{k:<22} {v if v is not None else 'not found'}")


if __name__ == "__main__":
    main()
