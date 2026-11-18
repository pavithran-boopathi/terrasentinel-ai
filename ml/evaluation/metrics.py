"""Binary segmentation metrics for the flood (positive) class with an ignore index.

All metrics are computed from a confusion matrix accumulated over pixels (``pooled``), which is
the standard for Sen1Floods11. Per-chip IoU is also kept for distributions and error analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

IGNORE_INDEX = 255


def confusion(pred: torch.Tensor, target: torch.Tensor, ignore_index: int = IGNORE_INDEX):
    """pred, target: (..., H, W) with values {0,1} (target may hold ``ignore_index``).
    Returns (tp, fp, fn, tn) as python ints."""
    valid = target != ignore_index
    p = pred.bool() & valid
    t = (target == 1) & valid
    tp = int((p & t).sum())
    fp = int((p & ~t).sum())
    fn = int((~p & t & valid).sum())
    tn = int((~p & ~t & valid).sum())
    return tp, fp, fn, tn


def _safe(n: float, d: float) -> float:
    return float(n / d) if d else float("nan")


def scores(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    precision = _safe(tp, tp + fp)
    recall = _safe(tp, tp + fn)
    return {
        "iou": _safe(tp, tp + fp + fn),
        "dice": _safe(2 * tp, 2 * tp + fp + fn),  # == F1 for binary
        "precision": precision,
        "recall": recall,
        "f1": _safe(2 * tp, 2 * tp + fp + fn),
        "accuracy": _safe(tp + tn, tp + fp + fn + tn),
        "iou_background": _safe(tn, tn + fp + fn),
    }


@dataclass
class MetricAccumulator:
    ignore_index: int = IGNORE_INDEX
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    per_chip: dict[str, dict[str, float]] = field(default_factory=dict)

    def update(self, pred: torch.Tensor, target: torch.Tensor, chips: list[str] | None = None):
        """pred/target: (B, H, W). Chip-level stats are recorded when ``chips`` is given."""
        pred, target = pred.detach().cpu(), target.detach().cpu()
        for i in range(pred.shape[0]):
            c = confusion(pred[i], target[i], self.ignore_index)
            self.tp += c[0]
            self.fp += c[1]
            self.fn += c[2]
            self.tn += c[3]
            if chips is not None:
                s = scores(*c)
                s.update(tp=c[0], fp=c[1], fn=c[2], tn=c[3])
                s["n_water"] = c[0] + c[2]
                s["n_valid"] = sum(c)
                self.per_chip[chips[i]] = s

    def compute(self) -> dict[str, float]:
        out = scores(self.tp, self.fp, self.fn, self.tn)
        ious = [v["iou"] for v in self.per_chip.values() if not np.isnan(v["iou"])]
        if ious:
            out["mean_chip_iou"] = float(np.mean(ious))
        return out


def bootstrap_iou_ci(
    per_chip_counts: list[tuple[int, int, int]], n_boot: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """95% CI of pooled IoU by resampling chips (tp, fp, fn). Chips, not pixels, are the
    independent unit — resampling pixels would give far too narrow intervals."""
    if not per_chip_counts:
        return float("nan"), float("nan")
    arr = np.asarray(per_chip_counts, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    s = arr[idx].sum(axis=1)
    iou = s[:, 0] / np.maximum(s.sum(axis=1), 1)
    lo, hi = np.percentile(iou, [2.5, 97.5])
    return float(lo), float(hi)


def group_scores(per_chip: dict[str, dict[str, float]], key=lambda chip: chip.split("_")[0]):
    """Pooled scores per group (default: flood event = chip-name prefix) with chip-bootstrap CI."""
    groups: dict[str, list[dict[str, float]]] = {}
    for chip, s in per_chip.items():
        groups.setdefault(key(chip), []).append(s)
    out = {}
    for g, rows in sorted(groups.items()):
        tp, fp, fn, tn = (int(sum(r[k] for r in rows)) for k in ("tp", "fp", "fn", "tn"))
        sc = scores(tp, fp, fn, tn)
        sc["iou_ci95"] = bootstrap_iou_ci([(r["tp"], r["fp"], r["fn"]) for r in rows])
        sc["chips"] = len(rows)
        out[g] = sc
    return out
