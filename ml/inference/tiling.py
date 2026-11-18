"""Sliding-window inference for arbitrarily large rasters.

Tiles of ``tile`` px are read with ``overlap`` px on each side, predicted in batches, and blended
with a 2-D raised-cosine (Hann) window so seams between tiles disappear. Accumulators are
memory-mapped float32 files, so peak RAM stays ~O(batch × tile²) regardless of scene size.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TileSpec:
    row: int  # top-left of the tile in image coords
    col: int
    height: int  # valid (in-image) extent; tiles at the border are padded to ``tile``
    width: int


def tile_grid(height: int, width: int, tile: int, overlap: int) -> list[TileSpec]:
    """Cover the image with tiles of size ``tile`` and stride ``tile - 2*overlap``.
    The last row/column is shifted inward so every tile lies fully inside the image when the
    image is at least one tile large."""
    if tile <= 2 * overlap:
        raise ValueError("tile must be larger than 2*overlap")
    stride = tile - 2 * overlap

    def starts(n: int) -> list[int]:
        if n <= tile:
            return [0]
        s = list(range(0, n - tile, stride))
        s.append(n - tile)
        return sorted(set(s))

    return [
        TileSpec(r, c, min(tile, height - r), min(tile, width - c))
        for r in starts(height)
        for c in starts(width)
    ]


def _ramp(overlap: int) -> np.ndarray:
    """Weights for the ``overlap`` px next to an interior tile edge: the outer quarter is exactly
    0 (those pixels see zero padding / truncated context), the rest rises as a raised cosine."""
    dead = overlap // 4
    rise = overlap - dead
    r = np.zeros(overlap, np.float32)
    if rise > 0:
        r[dead:] = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, rise + 2, dtype=np.float32)[1:-1])
    return r


def hann_window(
    tile: int,
    overlap: int,
    *,
    top: bool = True,
    bottom: bool = True,
    left: bool = True,
    right: bool = True,
) -> np.ndarray:
    """2-D blending weights. Sides flagged False touch the image boundary and are not tapered
    (no other tile covers those pixels)."""

    def axis(lo: bool, hi: bool) -> np.ndarray:
        w = np.ones(tile, np.float32)
        if overlap > 0:
            r = _ramp(overlap)
            if lo:
                w[:overlap] = r
            if hi:
                w[-overlap:] = r[::-1]
        return w

    return np.outer(axis(top, bottom), axis(left, right))


def iter_batches(specs: list[TileSpec], batch: int) -> Iterator[list[TileSpec]]:
    for i in range(0, len(specs), batch):
        yield specs[i : i + batch]


def tiled_predict(
    read_tile: Callable[[TileSpec, int], np.ndarray],
    predict_batch: Callable[[np.ndarray], np.ndarray],
    height: int,
    width: int,
    *,
    tile: int = 512,
    overlap: int = 64,
    batch: int = 4,
    workdir: str | Path | None = None,
    progress: Callable[[float], None] | None = None,
) -> np.ndarray:
    """Return a (height, width) float32 memmap of blended predictions.

    ``read_tile(spec, tile)`` returns a (C, tile, tile) float32 array (zero-padded at borders);
    ``predict_batch(x)`` maps (B, C, tile, tile) → (B, tile, tile) probabilities.
    """
    specs = tile_grid(height, width, tile, overlap)
    windows: dict[tuple[bool, bool, bool, bool], np.ndarray] = {}
    tmp = Path(workdir or tempfile.mkdtemp(prefix="ts_tiles_"))
    tmp.mkdir(parents=True, exist_ok=True)
    acc = np.memmap(tmp / "acc.f32", np.float32, "w+", shape=(height, width))
    wsum = np.memmap(tmp / "wsum.f32", np.float32, "w+", shape=(height, width))
    done = 0
    for group in iter_batches(specs, batch):
        x = np.stack([read_tile(s, tile) for s in group]).astype(np.float32)
        p = predict_batch(x)
        for s, pi in zip(group, p, strict=True):
            h, w = s.height, s.width
            key = (s.row > 0, s.row + tile < height, s.col > 0, s.col + tile < width)
            if key not in windows:
                windows[key] = hann_window(
                    tile, overlap, top=key[0], bottom=key[1], left=key[2], right=key[3]
                )
            win = windows[key]
            acc[s.row : s.row + h, s.col : s.col + w] += pi[:h, :w] * win[:h, :w]
            wsum[s.row : s.row + h, s.col : s.col + w] += win[:h, :w]
        done += len(group)
        if progress:
            progress(done / len(specs))
    out = np.memmap(tmp / "out.f32", np.float32, "w+", shape=(height, width))
    np.divide(acc, np.maximum(wsum, 1e-12), out=out)
    out.flush()
    del acc, wsum
    return out
