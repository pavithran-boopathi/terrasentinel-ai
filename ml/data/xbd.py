"""xBD / xView2 building damage — label parsing, event-level splits, Dataset, xView2 metric.

This is a *separate task* from flood segmentation: it classifies structural damage of buildings
from pre/post very-high-resolution optical imagery. It never consumes flood masks.

Expected layout (as distributed after registration at xview2.org)::

    <root>/<split>/images/<disaster>_<id>_{pre,post}_disaster.png   1024×1024 RGB
    <root>/<split>/labels/<disaster>_<id>_post_disaster.json         WKT polygons in pixel space

Damage classes: 0 background, 1 no-damage, 2 minor, 3 major, 4 destroyed;
``un-classified`` buildings are 255 (ignored).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from rasterio.features import rasterize
from shapely import wkt

DAMAGE = {"no-damage": 1, "minor-damage": 2, "major-damage": 3, "destroyed": 4}
CLASSES = ("background", "no-damage", "minor-damage", "major-damage", "destroyed")
IGNORE = 255


@dataclass(frozen=True)
class XBDTile:
    disaster: str
    tile_id: str
    pre: Path
    post: Path
    label: Path


def index_tiles(root: Path) -> list[XBDTile]:
    out = []
    for lbl in sorted(root.rglob("labels/*_post_disaster.json")):
        stem = lbl.name.removesuffix("_post_disaster.json")
        disaster, _, tile_id = stem.rpartition("_")
        img_dir = lbl.parent.parent / "images"
        pre, post = img_dir / f"{stem}_pre_disaster.png", img_dir / f"{stem}_post_disaster.png"
        if pre.exists() and post.exists():
            out.append(XBDTile(disaster, tile_id, pre, post, lbl))
    return out


def rasterize_damage(label_json: dict[str, Any], shape: tuple[int, int]) -> np.ndarray:
    """Pixel-space WKT polygons → uint8 damage mask (painted in increasing severity)."""
    mask = np.zeros(shape, np.uint8)
    feats = label_json.get("features", {}).get("xy", [])
    order = sorted(feats, key=lambda f: DAMAGE.get(f["properties"].get("subtype", ""), IGNORE))
    shapes = []
    for f in order:
        if f["properties"].get("feature_type", "building") != "building":
            continue
        val = DAMAGE.get(f["properties"].get("subtype", "un-classified"), IGNORE)
        shapes.append((wkt.loads(f["wkt"]), val))
    if shapes:
        rasterize(shapes, out=mask)
    return mask


def event_split(
    tiles: list[XBDTile], test_disasters: set[str], val_disasters: set[str]
) -> dict[str, list[XBDTile]]:
    """Hold out whole disasters (no tile of a test disaster is ever seen in training)."""
    out: dict[str, list[XBDTile]] = {"train": [], "val": [], "test": []}
    for t in tiles:
        k = (
            "test"
            if t.disaster in test_disasters
            else "val"
            if t.disaster in val_disasters
            else "train"
        )
        out[k].append(t)
    return out


class XBDDataset:
    MEAN = np.array([0.485, 0.456, 0.406] * 2, np.float32)[:, None, None]
    STD = np.array([0.229, 0.224, 0.225] * 2, np.float32)[:, None, None]

    def __init__(self, tiles: list[XBDTile], crop: int | None = None, seed: int = 0) -> None:
        self.tiles = tiles
        self.crop = crop
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, i: int) -> dict[str, Any]:
        import torch

        t = self.tiles[i]
        pre = np.asarray(Image.open(t.pre).convert("RGB")).transpose(2, 0, 1)
        post = np.asarray(Image.open(t.post).convert("RGB")).transpose(2, 0, 1)
        y = rasterize_damage(json.loads(t.label.read_text()), pre.shape[1:])
        x = (np.concatenate([pre, post]).astype(np.float32) / 255 - self.MEAN) / self.STD
        if self.crop:
            h, w = y.shape
            r = int(self.rng.integers(0, h - self.crop + 1))
            c = int(self.rng.integers(0, w - self.crop + 1))
            x, y = (
                x[:, r : r + self.crop, c : c + self.crop],
                y[r : r + self.crop, c : c + self.crop],
            )
            if self.rng.random() < 0.5:
                x, y = x[:, :, ::-1], y[:, ::-1]
        return {
            "image": torch.from_numpy(np.ascontiguousarray(x)),
            "mask": torch.from_numpy(np.ascontiguousarray(y).astype(np.int64)),
            "tile": f"{t.disaster}_{t.tile_id}",
        }


def xview2_score(conf: np.ndarray) -> dict[str, float]:
    """xView2 metric from a (5×5) pixel confusion matrix [true, pred] over non-ignored pixels.

    Localization F1: building (classes 1–4) vs background. Damage F1: per damage class, computed
    on pixels that are buildings in the ground truth; combined by harmonic mean.
    score = 0.3·F1_loc + 0.7·F1_damage (xView2 challenge definition)."""

    def f1(tp: float, fp: float, fn: float) -> float:
        return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0

    b = conf[1:, 1:].sum()
    loc = f1(b, conf[0, 1:].sum(), conf[1:, 0].sum())
    dmg = conf[1:, 1:]  # GT building pixels, predicted building class
    per = []
    for k in range(4):
        tp = dmg[k, k]
        per.append(f1(tp, dmg[:, k].sum() - tp, dmg[k, :].sum() - tp))
    hm = len(per) / sum(1 / max(p, 1e-6) for p in per)
    return {
        "f1_localization": loc,
        "f1_damage_harmonic": hm,
        **{f"f1_{CLASSES[k + 1]}": v for k, v in enumerate(per)},
        "xview2_score": 0.3 * loc + 0.7 * hm,
    }
