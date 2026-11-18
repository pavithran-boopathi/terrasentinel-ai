#!/usr/bin/env python3
"""Aggregate run results into docs/results/experiments.json for the web app's Model Laboratory.

    python scripts/export_experiments_json.py

Reads every data/checkpoints/*/results.json (segmentation runs, baselines, SpaceNet 8) and the
quiet-machine latency benchmark if present. Only numbers present in those files are exported.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CKPT = REPO / "data/checkpoints"
OUT = REPO / "docs/results/experiments.json"
EXCLUDE_RUNS = {"unet-official-20261118-143219"}  # user's manual rerun (reproducibility check)


def stats(vals: list[float]) -> dict[str, float | int]:
    a = np.array([v for v in vals if v == v], float)
    return {
        "n": int(a.size),
        "mean": float(a.mean()) if a.size else None,
        "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "min": float(a.min()) if a.size else None,
        "max": float(a.max()) if a.size else None,
    }


def main() -> None:
    seg: dict[tuple[str, str], list[dict]] = defaultdict(list)
    sn8 = []
    baselines = None
    for rf in sorted(CKPT.glob("*/results.json")):
        run = rf.parent.name
        if run in EXCLUDE_RUNS:
            continue
        r = json.loads(rf.read_text())
        if run.startswith("baselines-"):
            baselines = r
        elif run.startswith("sn8-"):
            sn8.append(
                {
                    "run": run,
                    "best_epoch": r["best_epoch"],
                    "test": r["test"],
                    "val": r["val"],
                    "baselines_test": r.get("baselines_test"),
                }
            )
        elif "splits" in r:
            cfg = r["config"]
            seg[(cfg.get("arch", "unet"), cfg["split_scheme"])].append(r)
    models = []
    for (arch, scheme), runs in sorted(seg.items()):
        splits = sorted({s for r in runs for s in r["splits"]})
        entry = {
            "arch": arch,
            "split_scheme": scheme,
            "seeds": [r["config"]["seed"] for r in runs],
            "params": runs[0]["meta"]["params"],
            "runs": [r["meta"]["run_name"] for r in runs],
            "splits": {},
        }
        for s in splits:
            rs = [r for r in runs if s in r["splits"]]
            entry["splits"][s] = {
                m: stats([r["splits"][s]["pooled"][m] for r in rs])
                for m in ("iou", "f1", "precision", "recall")
            }
            tuned = [
                r["splits_tuned_threshold"][s]["iou"]
                for r in rs
                if "splits_tuned_threshold" in r and s in r["splits_tuned_threshold"]
            ]
            if tuned:
                entry["splits"][s]["iou_val_tuned_threshold"] = stats(tuned)
            ev: dict[str, list[float]] = defaultdict(list)
            for r in rs:
                for e, v in r["splits"][s]["per_event"].items():
                    ev[e].append(v["iou"])
            entry["splits"][s]["per_event_iou"] = {e: stats(v) for e, v in ev.items()}
        models.append(entry)
    base_rows = []
    if baselines:
        for scheme, r in baselines.items():
            for method, per_split in r["methods"].items():
                for s, v in per_split.items():
                    p = v["pooled"]
                    base_rows.append(
                        {
                            "method": method,
                            "split_scheme": scheme,
                            "split": s,
                            "iou": p["iou"],
                            "f1": p["f1"],
                            "precision": p["precision"],
                            "recall": p["recall"],
                            "iou_ci95": p.get("iou_ci95"),
                        }
                    )
    lat_file = REPO / "docs/results/latency.json"
    out = {
        "generated": time.strftime("%Y-%m-%d"),
        "note": "Aggregated from data/checkpoints/*/results.json by scripts/export_experiments_json.py",
        "sen1floods11": {"models": models, "baselines": base_rows},
        "latency": json.loads(lat_file.read_text()) if lat_file.exists() else None,
        "spacenet8": sn8,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, default=float))
    print(
        f"wrote {OUT.relative_to(REPO)}: {len(models)} model groups, {len(base_rows)} baseline rows, "
        f"{len(sn8)} SpaceNet 8 runs"
    )


if __name__ == "__main__":
    main()
