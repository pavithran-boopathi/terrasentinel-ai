"""Probability calibration for binary segmentation: ECE, reliability curve, Brier, temperature
scaling. Pixels marked ``IGNORE_INDEX`` are excluded everywhere."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ml.evaluation.metrics import IGNORE_INDEX


@dataclass
class ReliabilityBins:
    edges: np.ndarray
    confidence: np.ndarray  # mean predicted P(water) per bin
    accuracy: np.ndarray  # observed water frequency per bin
    counts: np.ndarray


def reliability(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> ReliabilityBins:
    """Bins on P(water) (not max-class confidence): the natural view for a binary flood map."""
    p, y = _valid(probs, labels)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    counts = np.bincount(idx, minlength=n_bins)
    conf = np.bincount(idx, weights=p, minlength=n_bins) / np.maximum(counts, 1)
    acc = np.bincount(idx, weights=y, minlength=n_bins) / np.maximum(counts, 1)
    return ReliabilityBins(edges, conf, acc, counts)


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    r = reliability(probs, labels, n_bins)
    w = r.counts / max(1, r.counts.sum())
    return float(np.sum(w * np.abs(r.confidence - r.accuracy)))


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    p, y = _valid(probs, labels)
    return float(np.mean((p - y) ** 2))


def nll(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-6) -> float:
    p, y = _valid(probs, labels)
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _valid(probs: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probs, labels = np.asarray(probs).ravel(), np.asarray(labels).ravel()
    m = labels != IGNORE_INDEX
    return probs[m].astype(np.float64), (labels[m] == 1).astype(np.float64)


def fit_temperature(logits: np.ndarray, labels: np.ndarray, max_iter: int = 200) -> float:
    """Single scalar T > 0 minimising NLL of sigmoid(logits / T) on held-out (validation) pixels.
    Temperature scaling preserves the ranking of pixels, so IoU at a threshold of 0.5 changes
    only through the threshold's position, while calibration improves (Guo et al., 2017)."""
    z, y = _valid(logits, labels)
    zt = torch.tensor(z, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(zt / log_t.exp(), yt)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp())


def subsample(arrays: list[np.ndarray], labels: list[np.ndarray], per_chip: int, seed: int = 0):
    """Random valid pixels per chip, so calibration statistics fit in memory and every chip
    contributes equally."""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for a, y in zip(arrays, labels, strict=True):
        idx = np.flatnonzero(y.ravel() != IGNORE_INDEX)
        if idx.size == 0:
            continue
        pick = rng.choice(idx, size=min(per_chip, idx.size), replace=False)
        xs.append(a.ravel()[pick])
        ys.append(y.ravel()[pick])
    return np.concatenate(xs), np.concatenate(ys)
