"""Non-deep-learning baselines for SAR water segmentation.

* ``otsu_threshold``      — Otsu's method on a 1-D histogram (per chip, unsupervised)
* ``GlobalThreshold``     — single VH dB threshold chosen to maximise train IoU
* ``PixelClassifier``     — gradient-boosted trees on per-pixel backscatter + local texture
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter

from ml.data.sen1floods11 import IGNORE_INDEX


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Threshold maximising between-class variance. ``values`` must be finite, 1-D."""
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("no finite values")
    lo, hi = float(values.min()), float(values.max())
    if lo == hi:
        return lo
    hist, edges = np.histogram(values, bins=bins, range=(lo, hi))
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers)
    mu0 = m0 / np.maximum(w0, 1)
    mu1 = (m0[-1] - m0) / np.maximum(w1, 1)
    between = w0 * w1 * (mu0 - mu1) ** 2
    return float(centers[int(np.argmax(between))])


def otsu_predict(img_db: np.ndarray, band: int = 1) -> np.ndarray:
    """Water = backscatter below the chip's Otsu threshold (VH by default)."""
    x = img_db[band]
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros(x.shape, np.uint8)
    t = otsu_threshold(x[finite])
    return ((x < t) & finite).astype(np.uint8)


@dataclass
class GlobalThreshold:
    band: int = 1
    threshold: float | None = None

    def fit(self, images: list[np.ndarray], targets: list[np.ndarray]) -> GlobalThreshold:
        """Grid-search the dB threshold maximising pooled IoU on the given (train) chips."""
        grid = np.arange(-35.0, -5.0, 0.25)
        vals = np.concatenate(
            [im[self.band][t != IGNORE_INDEX] for im, t in zip(images, targets, strict=True)]
        )
        lab = np.concatenate([t[t != IGNORE_INDEX] for t in targets]) == 1
        ok = np.isfinite(vals)
        vals, lab = vals[ok], lab[ok]
        order = np.argsort(vals)
        v, y = vals[order], lab[order]
        cum_pos = np.cumsum(y)
        total_pos = cum_pos[-1]
        best, best_t = -1.0, grid[0]
        for t in grid:
            k = np.searchsorted(v, t)  # predicted water = first k pixels
            tp = cum_pos[k - 1] if k else 0
            fp = k - tp
            fn = total_pos - tp
            iou = tp / max(1, tp + fp + fn)
            if iou > best:
                best, best_t = iou, float(t)
        self.threshold = best_t
        return self

    def predict(self, img_db: np.ndarray) -> np.ndarray:
        if self.threshold is None:
            raise RuntimeError("fit first")
        x = img_db[self.band]
        return ((x < self.threshold) & np.isfinite(x)).astype(np.uint8)


def pixel_features(img_db: np.ndarray, windows: tuple[int, ...] = (5, 15)) -> np.ndarray:
    """(H*W, F) features: VV, VH, VV-VH, and local mean/std of VV and VH at each window size.
    NaNs are filled with the chip median before filtering so they don't spread."""
    x = np.clip(img_db.astype(np.float32), -50, 5)
    for b in range(x.shape[0]):
        band = x[b]
        if not np.isfinite(band).all():
            fill = np.nanmedian(band) if np.isfinite(band).any() else -20.0
            x[b] = np.where(np.isfinite(band), band, fill)
    vv, vh = x[0], x[1]
    feats = [vv, vh, vv - vh]
    for w in windows:
        for b in (vv, vh):
            m = uniform_filter(b, w)
            sq = uniform_filter(b * b, w)
            feats += [m, np.sqrt(np.maximum(sq - m * m, 0))]
    return np.stack(feats, axis=-1).reshape(-1, len(feats))


@dataclass
class PixelClassifier:
    samples_per_chip: int = 3000
    seed: int = 0
    model: object | None = None

    def fit(self, images: list[np.ndarray], targets: list[np.ndarray]) -> PixelClassifier:
        from sklearn.ensemble import HistGradientBoostingClassifier

        rng = np.random.default_rng(self.seed)
        xs, ys = [], []
        for im, t in zip(images, targets, strict=True):
            f = pixel_features(im)
            y = t.reshape(-1)
            valid = np.flatnonzero((y != IGNORE_INDEX) & np.isfinite(im).all(0).reshape(-1))
            if valid.size == 0:
                continue
            pick = rng.choice(valid, size=min(self.samples_per_chip, valid.size), replace=False)
            xs.append(f[pick])
            ys.append(y[pick])
        self.model = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.1, max_leaf_nodes=31, random_state=self.seed
        ).fit(np.concatenate(xs), np.concatenate(ys))
        return self

    def predict(self, img_db: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("fit first")
        p = self.model.predict_proba(pixel_features(img_db))[:, 1].reshape(img_db.shape[1:])
        return ((p > threshold) & np.isfinite(img_db).all(0)).astype(np.uint8)
