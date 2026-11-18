#!/usr/bin/env python3
"""Calibration (ECE/Brier/NLL ± temperature scaling) and uncertainty (MC dropout, TTA) of trained
runs on their test split(s). Temperature is fitted on the run's own validation split.

    python scripts/evaluate_uncertainty.py --runs data/checkpoints/<run> [...]

Writes docs/results/uncertainty.json and docs/figures/reliability_<run>.png.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from ml.data.sen1floods11 import NormStats, Sen1Floods11Dataset, load_splits_json  # noqa: E402
from ml.evaluation.calibration import (  # noqa: E402
    brier,
    ece,
    fit_temperature,
    nll,
    reliability,
)
from ml.evaluation.uncertainty import (  # noqa: E402
    error_detection_auroc,
    mc_dropout_probs,
    risk_coverage,
    summarize_samples,
    tta_probs,
)
from ml.models.factory import model_from_config  # noqa: E402
from ml.training.engine import select_device  # noqa: E402

RAW = REPO / "data/raw/sen1floods11"
PROC = REPO / "data/processed/sen1floods11"
PER_CHIP = 20_000


@torch.no_grad()
def collect(model, ds, device, mode: str, rng):
    """Subsampled per-pixel arrays: logit, label, and (for mc/tta) mean prob + uncertainty."""
    out = {"logit": [], "label": [], "prob_u": [], "unc": []}
    for i in range(len(ds)):
        b = ds[i]
        x = b["image"][None].to(device)
        y = b["mask"].numpy()
        idx = np.flatnonzero(y.ravel() != 255)
        if idx.size == 0:
            continue
        pick = rng.choice(idx, size=min(PER_CHIP, idx.size), replace=False)
        model.eval()
        z = model(x)[0, 0].float().cpu().numpy().ravel()[pick]
        out["logit"].append(z)
        out["label"].append(y.ravel()[pick])
        if mode in ("mc", "tta"):
            s = mc_dropout_probs(model, x, passes=10) if mode == "mc" else tta_probs(model, x)
            summ = summarize_samples(s.float().cpu())
            out["prob_u"].append(summ["mean"][0].numpy().ravel()[pick])
            out["unc"].append(summ["entropy"][0].numpy().ravel()[pick])
    return {k: np.concatenate(v) if v else None for k, v in out.items()}


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def plot_reliability(path: Path, curves: dict[str, tuple]):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], color="#888", lw=1, ls="--", label="perfect")
    for name, (conf, acc, cnt) in curves.items():
        m = cnt > 50
        ax.plot(conf[m], acc[m], marker="o", ms=3, label=name)
    ax.set_xlabel("predicted P(water)")
    ax.set_ylabel("observed water frequency")
    ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def evaluate_run(run: Path, device) -> dict:
    state = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    cfg = state["config"]
    model = model_from_config(cfg)
    model.load_state_dict(state["model"])
    model.to(device)
    has_dropout = any(isinstance(m, torch.nn.Dropout | torch.nn.Dropout2d) for m in model.modules())
    scheme = cfg["split_scheme"]
    splits = load_splits_json(PROC / "splits" / f"{scheme}.json")
    stats = NormStats.from_json((PROC / f"norm_stats_{scheme}.json").read_text())
    rng = np.random.default_rng(0)
    val = collect(model, Sen1Floods11Dataset(splits["val"], RAW, stats), device, "plain", rng)
    T = fit_temperature(val["logit"], val["label"])
    res = {
        "run": run.name,
        "arch": cfg.get("arch", "unet"),
        "split_scheme": scheme,
        "temperature": T,
        "mc_dropout_available": has_dropout,
        "splits": {},
    }
    curves = {}
    for split in cfg["eval_splits"]:
        ds = Sen1Floods11Dataset(splits[split], RAW, stats)
        d = collect(model, ds, device, "tta", np.random.default_rng(1))
        p, pT, y = sigmoid(d["logit"]), sigmoid(d["logit"] / T), d["label"]
        r = {
            "pixels": int(len(y)),
            "ece": ece(p, y),
            "ece_temperature_scaled": ece(pT, y),
            "brier": brier(p, y),
            "brier_temperature_scaled": brier(pT, y),
            "nll": nll(p, y),
            "nll_temperature_scaled": nll(pT, y),
            "tta_entropy_error_auroc": error_detection_auroc(d["unc"], d["prob_u"], y),
            "tta_risk_coverage_iou": risk_coverage(d["unc"], d["prob_u"], y),
        }
        if has_dropout:
            m = collect(model, ds, device, "mc", np.random.default_rng(1))
            r["mc_entropy_error_auroc"] = error_detection_auroc(m["unc"], m["prob_u"], y)
            r["mc_risk_coverage_iou"] = risk_coverage(m["unc"], m["prob_u"], y)
        rel, relT = reliability(p, y), reliability(pT, y)
        curves[f"{split} raw (ECE {r['ece']:.3f})"] = (rel.confidence, rel.accuracy, rel.counts)
        curves[f"{split} T={T:.2f} (ECE {r['ece_temperature_scaled']:.3f})"] = (
            relT.confidence,
            relT.accuracy,
            relT.counts,
        )
        res["splits"][split] = r
        print(
            run.name,
            split,
            {k: round(v, 4) for k, v in r.items() if isinstance(v, float)},
            flush=True,
        )
    figs = REPO / "docs/figures"
    figs.mkdir(parents=True, exist_ok=True)
    plot_reliability(figs / f"reliability_{res['arch']}_{scheme}.png", curves)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    a = ap.parse_args()
    device = select_device()
    out = [evaluate_run(r, device) for r in a.runs]
    dest = REPO / "docs/results/uncertainty.json"
    dest.write_text(json.dumps({"per_chip_pixels": PER_CHIP, "runs": out}, indent=1, default=float))
    print(f"wrote {dest.relative_to(REPO)}")


if __name__ == "__main__":
    main()
