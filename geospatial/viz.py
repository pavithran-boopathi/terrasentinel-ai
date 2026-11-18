"""Visualisation helpers for SAR chips and segmentation masks (pure numpy → uint8 RGB)."""

from __future__ import annotations

import numpy as np

# Display ranges in dB, typical for Sentinel-1 GRD backscatter.
VV_RANGE = (-25.0, 0.0)
VH_RANGE = (-30.0, -5.0)


def _stretch(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    y = (np.clip(x, lo, hi) - lo) / (hi - lo)
    return np.nan_to_num(y, nan=0.0)


def sar_false_color(vv_db: np.ndarray, vh_db: np.ndarray) -> np.ndarray:
    """Common S1 composite: R=VV, G=VH, B=VV-VH (ratio in dB). Returns (H, W, 3) uint8.
    Open water is dark in all channels; NaN pixels render black."""
    r = _stretch(vv_db, *VV_RANGE)
    g = _stretch(vh_db, *VH_RANGE)
    b = _stretch(vv_db - vh_db, 0.0, 15.0)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


LABEL_COLORS = {
    -1: (0, 0, 0, 0),  # no data: transparent
    0: (0, 0, 0, 0),  # not water: transparent
    1: (30, 144, 255, 170),  # water
}


def overlay_mask(
    rgb: np.ndarray, mask: np.ndarray, colors: dict[int, tuple[int, int, int, int]] | None = None
) -> np.ndarray:
    """Alpha-blend a categorical mask over an RGB image. Returns uint8 RGB."""
    colors = colors or LABEL_COLORS
    out = rgb.astype(np.float32)
    for value, (r, g, b, a) in colors.items():
        if a == 0:
            continue
        sel = mask == value
        alpha = a / 255
        out[sel] = out[sel] * (1 - alpha) + np.array([r, g, b], np.float32) * alpha
    return out.clip(0, 255).astype(np.uint8)


def nodata_hatch(rgb: np.ndarray, invalid: np.ndarray, color=(255, 0, 255)) -> np.ndarray:
    """Mark invalid pixels with a diagonal hatch so they are not confused with water."""
    out = rgb.copy()
    h, w = invalid.shape
    yy, xx = np.mgrid[0:h, 0:w]
    out[invalid & ((xx + yy) % 8 < 2)] = color
    return out
