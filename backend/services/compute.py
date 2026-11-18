"""Detect the compute device available for ML workloads.

Torch is an optional dependency (``.[ml]``); the API must work without it.
"""

from __future__ import annotations

import platform
import sys
from typing import Any


def compute_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "python_version": sys.version.split()[0],
        "torch_version": None,
        "mps_available": None,
        "cuda_available": None,
        "recommended_device": "cpu",
    }
    try:
        import torch
    except ImportError:
        return info
    info["torch_version"] = torch.__version__
    info["mps_available"] = bool(torch.backends.mps.is_available())
    info["cuda_available"] = bool(torch.cuda.is_available())
    if info["cuda_available"]:
        info["recommended_device"] = "cuda"
    elif info["mps_available"]:
        info["recommended_device"] = "mps"
    return info
