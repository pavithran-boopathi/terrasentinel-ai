"""Leave-one-event-out (LOEO) cross-validation on Sen1Floods11 hand-labeled chips.

    python -m ml.training.loeo --arch unet_resnet34 --epochs 35 [--events Bolivia Ghana ...]

For each of the 11 flood events: train on the other 10, test on the held-out event. There is no
validation-based checkpoint selection (a fixed epoch budget, chosen from Phase 3/4 best epochs),
so no information about the held-out event leaks into training. Normalisation statistics are
recomputed from each fold's training chips only. A gradient-boosted pixel baseline is evaluated on
the same folds for context.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ml.baselines.classical import PixelClassifier
from ml.data.sen1floods11 import (
    Sen1Floods11Dataset,
    compute_norm_stats,
    load_splits_json,
    read_chip,
)
from ml.evaluation.metrics import MetricAccumulator, bootstrap_iou_ci
from ml.models.factory import build_model
from ml.training.augment import train_transform
from ml.training.engine import (
    cosine_with_warmup,
    evaluate,
    seed_everything,
    select_device,
    train_one_epoch,
)
from ml.training.losses import BCEDiceLoss
from ml.training.train import git_state

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data/raw/sen1floods11"
LOEO_DIR = REPO / "data/processed/sen1floods11/splits/loeo"


def run_fold(event: str, arch: str, epochs: int, seed: int, device) -> dict[str, Any]:
    split = load_splits_json(LOEO_DIR / f"{event}.json")
    stats = compute_norm_stats(split["train"], RAW, source=f"loeo/{event}/train")
    train = Sen1Floods11Dataset(split["train"], RAW, stats, transform=train_transform(256, 0.05))
    test = Sen1Floods11Dataset(split["test"], RAW, stats)
    train.preload()
    test.preload()
    water = dry = 0
    for _, t in train._cache:
        water += int((t == 1).sum())
        dry += int((t == 0).sum())
    pos_weight = float(np.sqrt(dry / max(1, water)))
    seed_everything(seed)
    kw = {"base_channels": 32, "depth": 4, "dropout": 0.1} if arch == "unet" else {"dropout": 0.1}
    model = build_model(arch, pretrained=arch != "unet", **kw).to(device)
    loss_fn = BCEDiceLoss(pos_weight=pos_weight).to(device)
    lr = 1e-3 if arch == "unet" else 3e-4 if arch == "unet_resnet34" else 2e-4
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    dl = DataLoader(
        train, 8, shuffle=True, generator=torch.Generator().manual_seed(seed), drop_last=True
    )
    sched = cosine_with_warmup(opt, 3 * len(dl), epochs * len(dl))
    t0 = time.perf_counter()
    for _ in range(epochs):
        train_one_epoch(model, dl, loss_fn, opt, sched, device, grad_clip=1.0)
    m, acc = evaluate(model, DataLoader(test, 8), device, keep_per_chip=True)
    m["iou_ci95"] = bootstrap_iou_ci([(v["tp"], v["fp"], v["fn"]) for v in acc.per_chip.values()])

    # Same fold, gradient-boosted pixel baseline (fit on the same training chips).
    imgs = [read_chip(c, RAW) for c in split["train"]]
    pc = PixelClassifier().fit([i for i, _ in imgs], [t for _, t in imgs])
    bacc = MetricAccumulator()
    for c in split["test"]:
        img, t = read_chip(c, RAW)
        bacc.update(
            torch.from_numpy(pc.predict(img))[None], torch.from_numpy(t.astype(np.int64))[None]
        )
    return {
        "event": event,
        "train_chips": len(split["train"]),
        "test_chips": len(split["test"]),
        "pos_weight": pos_weight,
        "train_seconds": time.perf_counter() - t0,
        "model": m,
        "pixel_hgb": bacc.compute(),
        "test_water_fraction": (acc.tp + acc.fn) / max(1, acc.tp + acc.fn + acc.fp + acc.tn),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="unet_resnet34")
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--events", nargs="*")
    a = ap.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    device = select_device()
    events = a.events or sorted(p.stem for p in LOEO_DIR.glob("*.json"))
    out = REPO / "data/checkpoints" / f"loeo-{a.arch}-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True)
    folds = []
    git = git_state()  # captured once, at start
    for ev in events:
        r = run_fold(ev, a.arch, a.epochs, a.seed, device)
        folds.append(r)
        print(
            f"{ev:10s} U-Net IoU {r['model']['iou']:.3f}  pixel-GBT IoU "
            f"{r['pixel_hgb']['iou']:.3f}  ({r['train_seconds']:.0f}s)",
            flush=True,
        )
        (out / "results.json").write_text(
            json.dumps(
                {
                    "arch": a.arch,
                    "epochs": a.epochs,
                    "seed": a.seed,
                    "git": git,
                    "device": device.type,
                    "folds": folds,
                },
                indent=1,
                default=float,
            )
        )
    ious = np.array([f["model"]["iou"] for f in folds])
    base = np.array([f["pixel_hgb"]["iou"] for f in folds])
    print(
        json.dumps(
            {
                "mean_iou": ious.mean(),
                "std": ious.std(ddof=1),
                "min": ious.min(),
                "pixel_hgb_mean": base.mean(),
                "model_wins": int((ious > base).sum()),
                "folds": len(folds),
            },
            default=float,
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
