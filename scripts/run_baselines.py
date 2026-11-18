#!/usr/bin/env python3
"""Fit and evaluate classical baselines on every split of both split schemes.

python scripts/run_baselines.py            # writes data/checkpoints/baselines-<ts>/results.json
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ml.baselines.classical import GlobalThreshold, PixelClassifier, otsu_predict  # noqa: E402
from ml.data.sen1floods11 import HAND, load_splits_json, read_chip  # noqa: E402
from ml.evaluation.metrics import MetricAccumulator, bootstrap_iou_ci, group_scores  # noqa: E402

RAW = REPO / "data/raw/sen1floods11"
PROC = REPO / "data/processed/sen1floods11"


def dataset_otsu(chip) -> np.ndarray:
    with rasterio.open(RAW / HAND / "S1OtsuLabelHand" / f"{chip.name}_S1OtsuLabelHand.tif") as s:
        return (s.read(1) == 1).astype(np.uint8)


def evaluate(predict, chips, cache) -> dict:
    acc = MetricAccumulator()
    t0 = time.perf_counter()
    for c in chips:
        img, t = cache[c.name]
        acc.update(
            torch.from_numpy(predict(c, img))[None],
            torch.from_numpy(t.astype(np.int64))[None],
            [c.name],
        )
    ms = (time.perf_counter() - t0) * 1000 / max(1, len(chips))
    pooled = acc.compute()
    pooled["iou_ci95"] = bootstrap_iou_ci(
        [(v["tp"], v["fp"], v["fn"]) for v in acc.per_chip.values()]
    )
    pooled["ms_per_chip_cpu"] = ms
    return {"pooled": pooled, "per_event": group_scores(acc.per_chip), "per_chip": acc.per_chip}


def main() -> None:
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    out = REPO / "data/checkpoints" / f"baselines-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True)
    results: dict = {}
    for scheme, evals in {
        "official": ["val", "test", "bolivia"],
        "geo_holdout_v1": ["val", "test"],
    }.items():
        splits = load_splits_json(PROC / "splits" / f"{scheme}.json")
        cache = {c.name: read_chip(c, RAW) for v in splits.values() for c in v}
        train_imgs = [cache[c.name][0] for c in splits["train"]]
        train_tgts = [cache[c.name][1] for c in splits["train"]]
        t0 = time.perf_counter()
        gt = GlobalThreshold().fit(train_imgs, train_tgts)
        pc = PixelClassifier().fit(train_imgs, train_tgts)
        fit_s = time.perf_counter() - t0
        methods = {
            "otsu_vh_per_chip": lambda c, img: otsu_predict(img, band=1),
            "dataset_s1_otsu_labels": lambda c, img: dataset_otsu(c),
            f"global_vh_threshold({gt.threshold:.2f}dB)": lambda c, img, gt=gt: gt.predict(img),
            "pixel_hgb": lambda c, img, pc=pc: pc.predict(img),
        }
        results[scheme] = {"fit_seconds": fit_s, "vh_threshold_db": gt.threshold, "methods": {}}
        for mname, fn in methods.items():
            results[scheme]["methods"][mname] = {s: evaluate(fn, splits[s], cache) for s in evals}
            summary = {
                s: round(r["pooled"]["iou"], 4)
                for s, r in results[scheme]["methods"][mname].items()
            }
            print(f"{scheme:15s} {mname:32s} IoU {summary}", flush=True)
    (out / "results.json").write_text(json.dumps(results, indent=1, default=float))

    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{REPO / 'data/mlflow/mlflow.db'}")
    mlflow.set_experiment("sen1floods11-segmentation")
    for scheme, r in results.items():
        for mname, per_split in r["methods"].items():
            with mlflow.start_run(run_name=f"baseline-{mname}-{scheme}"):
                mlflow.log_params({"method": mname, "split_scheme": scheme, "kind": "baseline"})
                for s, v in per_split.items():
                    mlflow.log_metrics(
                        {
                            f"{s}_{k}": x
                            for k, x in v["pooled"].items()
                            if isinstance(x, float) and not np.isnan(x)
                        }
                    )
    print(f"results → {out.relative_to(REPO)}/results.json")


if __name__ == "__main__":
    main()
