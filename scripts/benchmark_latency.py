#!/usr/bin/env python3
"""Inference latency for every architecture on a quiet machine (run nothing else meanwhile).

    python scripts/benchmark_latency.py

Per-chip forward latency (512×512×2, synchronised, 10 warm-up + 50 timed iterations) on MPS
(if available) and CPU at batch 1 and 8, plus end-to-end tiled-inference throughput on a
synthetic 4096×4096 scene. Writes docs/results/latency.json.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from ml.inference.tiling import tiled_predict  # noqa: E402
from ml.models.factory import ARCHS, build_model  # noqa: E402
from ml.models.unet import count_parameters  # noqa: E402
from ml.training.engine import benchmark_latency  # noqa: E402


def main() -> None:
    torch.set_num_threads(os.cpu_count() or 4)
    devices = (["mps"] if torch.backends.mps.is_available() else []) + ["cpu"]
    rows = []
    for arch in ARCHS:
        kw = {"base_channels": 32, "depth": 4, "dropout": 0.1} if arch == "unet" else {}
        model = build_model(arch, pretrained=False, **kw)
        params = count_parameters(model)
        for dev in devices:
            d = torch.device(dev)
            m = model.to(d)
            for bs in (1, 8):
                r = benchmark_latency(
                    m, d, (bs, 2, 512, 512), warmup=10, iters=50 if bs == 1 else 15
                )
                rows.append(
                    {
                        "arch": arch,
                        "device": dev,
                        "batch": bs,
                        "params": params,
                        "ms_per_batch": r["latency_ms_mean"],
                        "ms_per_chip": r["latency_ms_mean"] / bs,
                        "p95_ms_per_batch": r["latency_ms_p95"],
                    }
                )
                print(rows[-1], flush=True)
    # Tiled throughput on a large scene with the fastest accurate model family (MPS only).
    tiled = []
    if "mps" in devices:
        size, tile, overlap = 4096, 512, 64
        img = np.random.default_rng(0).normal(size=(2, size, size)).astype(np.float32)

        def reader(s, t):
            out = np.zeros((2, t, t), np.float32)
            out[:, : s.height, : s.width] = img[
                :, s.row : s.row + s.height, s.col : s.col + s.width
            ]
            return out

        for arch in ("unet", "segformer_b0"):
            kw = {"base_channels": 32, "depth": 4, "dropout": 0.1} if arch == "unet" else {}
            m = build_model(arch, pretrained=False, **kw).to("mps").eval()

            @torch.no_grad()
            def pred(x, m=m):
                return torch.sigmoid(m(torch.from_numpy(x).to("mps"))[:, 0]).cpu().numpy()

            t0 = time.perf_counter()
            tiled_predict(
                reader,
                pred,
                size,
                size,
                tile=tile,
                overlap=overlap,
                batch=8,
                workdir=REPO / "data/tmp_bench",
            )
            sec = time.perf_counter() - t0
            km2 = (size * 10 / 1000) ** 2  # at Sentinel-1's 10 m pixels
            tiled.append(
                {
                    "arch": arch,
                    "scene_px": f"{size}x{size}",
                    "tile": tile,
                    "overlap": overlap,
                    "seconds": sec,
                    "km2_per_second_at_10m": km2 / sec,
                }
            )
            print(tiled[-1], flush=True)
    out = {
        "device": "mps" if "mps" in devices else "cpu",
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "rows": [r for r in rows if r["device"] == ("mps" if "mps" in devices else "cpu")],
        "all_rows": rows,
        "tiled": tiled,
        "date": time.strftime("%Y-%m-%d"),
    }
    dest = REPO / "docs/results/latency.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"wrote {dest.relative_to(REPO)}")


if __name__ == "__main__":
    main()
