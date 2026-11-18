"""Multitemporal SAR change detection for flood mapping.

* ``log_ratio``   — post − pre backscatter in dB (= log-ratio of intensities). Flooding of
                    land usually *decreases* backscatter (specular reflection off open water).
* ``new_water``   — water in the post scene that was not water in the pre scene (from any two
                    water masks, e.g. the segmentation model run on both dates).
* ``split_permanent`` — separates water that is normally present (JRC Global Surface Water
                    occurrence ≥ threshold) from flood water.
Masks use 1 = true, 0 = false, ``nodata`` (255) = unknown; unknown propagates.
"""

from __future__ import annotations

import numpy as np

NODATA = 255


def log_ratio(pre_db: np.ndarray, post_db: np.ndarray) -> np.ndarray:
    """dB difference per band; NaN where either date is missing."""
    pre, post = np.asarray(pre_db, np.float32), np.asarray(post_db, np.float32)
    if pre.shape != post.shape:
        raise ValueError(f"pre {pre.shape} and post {post.shape} must be co-registered")
    return post - pre


def darkening_mask(
    pre_db: np.ndarray,
    post_db: np.ndarray,
    *,
    band: int = 1,
    drop_db: float = 3.0,
    water_db: float = -18.0,
) -> np.ndarray:
    """Classic unsupervised rule: backscatter dropped by ≥ ``drop_db`` AND the post value is
    water-dark (VH below ``water_db``). Returns uint8 mask with NODATA where undefined."""
    d = log_ratio(pre_db, post_db)[band]
    post = np.asarray(post_db, np.float32)[band]
    out = ((d <= -drop_db) & (post <= water_db)).astype(np.uint8)
    out[~np.isfinite(d)] = NODATA
    return out


def new_water(pre_mask: np.ndarray, post_mask: np.ndarray) -> np.ndarray:
    pre, post = np.asarray(pre_mask), np.asarray(post_mask)
    if pre.shape != post.shape:
        raise ValueError("masks must be co-registered")
    out = ((post == 1) & (pre == 0)).astype(np.uint8)
    out[(pre == NODATA) | (post == NODATA)] = NODATA
    return out


def receded_water(pre_mask: np.ndarray, post_mask: np.ndarray) -> np.ndarray:
    return new_water(post_mask, pre_mask)


def split_permanent(
    water_mask: np.ndarray, occurrence_pct: np.ndarray, permanent_threshold: float = 50.0
) -> tuple[np.ndarray, np.ndarray]:
    """(flood_water, permanent_water) from a water mask and JRC occurrence (0–100 %)."""
    w = np.asarray(water_mask)
    occ = np.asarray(occurrence_pct, np.float32)
    permanent = occ >= permanent_threshold
    flood = ((w == 1) & ~permanent).astype(np.uint8)
    perm = ((w == 1) & permanent).astype(np.uint8)
    flood[w == NODATA] = NODATA
    perm[w == NODATA] = NODATA
    return flood, perm
