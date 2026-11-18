"""Predictive uncertainty for segmentation: MC dropout, test-time augmentation (TTA), and
diagnostics that ask whether uncertainty actually points at errors."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ml.evaluation.metrics import IGNORE_INDEX

DIHEDRAL = [(k, f) for k in range(4) for f in (False, True)]


def enable_mc_dropout(model: nn.Module) -> int:
    """Eval mode everywhere (BatchNorm uses running stats) but dropout layers sampling."""
    model.eval()
    n = 0
    for m in model.modules():
        if isinstance(m, nn.Dropout | nn.Dropout2d | nn.Dropout3d):
            m.train()
            n += 1
    return n


@torch.no_grad()
def mc_dropout_probs(model: nn.Module, x: torch.Tensor, passes: int = 10) -> torch.Tensor:
    """(passes, B, H, W) water probabilities with dropout active. Raises if the model has no
    dropout (MC dropout would silently return identical samples)."""
    if enable_mc_dropout(model) == 0:
        raise ValueError("model has no dropout layers; MC dropout is undefined")
    out = torch.stack([torch.sigmoid(model(x)[:, 0]) for _ in range(passes)])
    model.eval()
    return out


@torch.no_grad()
def tta_probs(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """(8, B, H, W) probabilities under the 8 dihedral transforms, mapped back to input frame."""
    model.eval()
    outs = []
    for k, flip in DIHEDRAL:
        t = torch.rot90(x, k, dims=(-2, -1))
        if flip:
            t = torch.flip(t, dims=(-1,))
        p = torch.sigmoid(model(t)[:, 0])
        if flip:
            p = torch.flip(p, dims=(-1,))
        outs.append(torch.rot90(p, -k, dims=(-2, -1)))
    return torch.stack(outs)


def summarize_samples(samples: torch.Tensor) -> dict[str, torch.Tensor]:
    """Mean probability, sample std, and binary predictive entropy (bits) of the mean."""
    mean = samples.mean(0)
    p = mean.clamp(1e-6, 1 - 1e-6)
    entropy = -(p * torch.log2(p) + (1 - p) * torch.log2(1 - p))
    return {"mean": mean, "std": samples.std(0), "entropy": entropy}


def error_detection_auroc(
    uncertainty: np.ndarray, prob: np.ndarray, labels: np.ndarray, threshold: float = 0.5
) -> float:
    """AUROC of the uncertainty score for separating misclassified from correct pixels.
    0.5 = uncertainty is uninformative about errors; 1.0 = perfectly ranks errors first."""
    u, p, y = (np.asarray(a).ravel() for a in (uncertainty, prob, labels))
    m = y != IGNORE_INDEX
    u, err = u[m], ((p[m] > threshold) != (y[m] == 1))
    n_pos, n_neg = int(err.sum()), int((~err).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(u, kind="mergesort")
    ranks = np.empty(len(u), np.float64)
    ranks[order] = np.arange(1, len(u) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(u, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / cnt)[inv]
    return float((ranks[err].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def risk_coverage(
    uncertainty: np.ndarray,
    prob: np.ndarray,
    labels: np.ndarray,
    coverages=(1.0, 0.95, 0.9, 0.8),
    threshold: float = 0.5,
) -> dict[float, float]:
    """Flood IoU restricted to the most-confident ``coverage`` fraction of valid pixels.
    If uncertainty is useful, IoU should rise as low-confidence pixels are deferred."""
    u, p, y = (np.asarray(a).ravel() for a in (uncertainty, prob, labels))
    m = y != IGNORE_INDEX
    u, pred, gt = u[m], p[m] > threshold, y[m] == 1
    order = np.argsort(u)
    out = {}
    for c in coverages:
        keep = order[: max(1, int(round(c * len(order))))]
        tp = np.sum(pred[keep] & gt[keep])
        fp = np.sum(pred[keep] & ~gt[keep])
        fn = np.sum(~pred[keep] & gt[keep])
        out[c] = float(tp / max(1, tp + fp + fn))
    return out
