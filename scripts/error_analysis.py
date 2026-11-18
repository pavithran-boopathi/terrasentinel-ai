#!/usr/bin/env python3
"""Error analysis for trained runs vs. baselines.

    python scripts/error_analysis.py --baselines data/checkpoints/baselines-*/results.json \
        --runs data/checkpoints/unet-official-*/ data/checkpoints/unet-geo-holdout-*/

Writes
  docs/figures/per_event_iou_<scheme>.png    per-event IoU, U-Net vs best baselines (committed)
  data/processed/error_analysis/<run>_worst.png   worst test chips: S1 / label / errors (not committed)
and prints a short text summary (worst chips, IoU vs water fraction correlation).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from geospatial.viz import sar_false_color  # noqa: E402
from ml.data.sen1floods11 import (  # noqa: E402
    IGNORE_INDEX,
    Chip,
    NormStats,
    normalize,
    read_chip,
)
from ml.models.unet import UNet  # noqa: E402
from ml.training.engine import select_device  # noqa: E402

RAW = REPO / "data/raw/sen1floods11"
PROC = REPO / "data/processed/sen1floods11"
BASELINES_SHOWN = ("pixel_hgb", "global_vh_threshold")


def per_event_chart(scheme: str, run: dict, baselines: dict, split: str, out: Path) -> None:
    ev_unet = run["splits"][split]["per_event"]
    events = list(ev_unet)
    series = {"U-Net": [ev_unet[e]["iou"] for e in events]}
    for name, per_split in baselines[scheme]["methods"].items():
        if name.startswith(BASELINES_SHOWN) and split in per_split:
            series[name.split("(")[0]] = [per_split[split]["per_event"][e]["iou"] for e in events]
    x = np.arange(len(events))
    w = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(events) + 2), 3.6))
    for i, (name, vals) in enumerate(series.items()):
        ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=name)
    ax.set_xticks(x, [f"{e}\n(n={ev_unet[e]['chips']})" for e in events], fontsize=8)
    ax.set_ylabel("Flood IoU (pooled)")
    ax.set_ylim(0, 1)
    ax.set_title(f"Per-event IoU — {scheme} / {split}", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


@torch.no_grad()
def worst_chips_figure(run_dir: Path, split: str, k: int, out: Path) -> list[tuple[str, float]]:
    res = json.loads((run_dir / "results.json").read_text())
    per_chip = res["splits"][split]["per_chip"]
    scored = [(c, v["iou"], v["n_water"]) for c, v in per_chip.items() if v["n_water"] > 500]
    scored.sort(key=lambda t: t[1])
    worst = scored[:k]
    state = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    cfg = state["config"]
    device = select_device()
    model = UNet(2, 1, cfg["base_channels"], cfg["depth"], cfg["dropout"])
    model.load_state_dict(state["model"])
    model.to(device).eval()
    stats = NormStats.from_json((PROC / f"norm_stats_{cfg['split_scheme']}.json").read_text())
    fig, axes = plt.subplots(len(worst), 3, figsize=(9, 3 * len(worst)))
    for row, (name, iou, _) in zip(np.atleast_2d(axes), worst, strict=True):
        chip = Chip.from_name(name)
        img, t = read_chip(chip, RAW)
        x = torch.from_numpy(normalize(img, stats))[None].to(device)
        pred = (torch.sigmoid(model(x))[0, 0] > cfg["threshold"]).cpu().numpy()
        rgb = sar_false_color(img[0], img[1])
        valid = t != IGNORE_INDEX
        err = np.zeros((*t.shape, 3), np.uint8)
        err[(pred == 1) & (t == 1)] = (40, 120, 255)  # TP blue
        err[(pred == 1) & (t == 0)] = (230, 40, 40)  # FP red
        err[(pred == 0) & (t == 1)] = (255, 200, 0)  # FN yellow
        err[~valid] = (90, 90, 90)
        lab = np.where(t == 1, 255, np.where(valid, 0, 120)).astype(np.uint8)
        for ax, im, title in zip(
            row,
            [rgb, lab, err],
            [f"{name} S1", "label (grey = ignore)", f"IoU {iou:.2f}: TP/FP/FN"],
            strict=True,
        ):
            ax.imshow(im, cmap="gray" if im.ndim == 2 else None)
            ax.set_title(title, fontsize=8)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=80)
    plt.close(fig)
    return [(n, i) for n, i, _ in worst]


def water_fraction_correlation(run: dict, split: str) -> float:
    rows = [v for v in run["splits"][split]["per_chip"].values() if v["n_valid"] > 0]
    wf = np.array([v["n_water"] / v["n_valid"] for v in rows])
    iou = np.array([v["iou"] for v in rows])
    ok = np.isfinite(iou)
    return float(np.corrcoef(wf[ok], iou[ok])[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", type=Path, required=True)
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()
    baselines = json.loads(a.baselines.read_text())
    figs = REPO / "docs/figures"
    figs.mkdir(parents=True, exist_ok=True)
    priv = REPO / "data/processed/error_analysis"
    priv.mkdir(parents=True, exist_ok=True)
    for run_dir in a.runs:
        run = json.loads((run_dir / "results.json").read_text())
        scheme = run["config"]["split_scheme"]
        for split in run["splits"]:
            if split == "val":
                continue
            per_event_chart(
                scheme, run, baselines, split, figs / f"per_event_iou_{scheme}_{split}.png"
            )
            r = water_fraction_correlation(run, split)
            worst = worst_chips_figure(
                run_dir, split, a.k, priv / f"{run_dir.name}_{split}_worst.png"
            )
            print(
                f"{run_dir.name} [{split}] corr(water fraction, chip IoU) = {r:.2f}; worst: "
                + ", ".join(f"{n} {i:.2f}" for n, i in worst)
            )


if __name__ == "__main__":
    main()
