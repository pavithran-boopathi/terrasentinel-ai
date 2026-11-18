"""SpaceNet 8 (flooded buildings & roads) — preprocessing, spatial splits, Dataset.

Each label tile has a pre-event image (RGB uint8, ~0.3–0.5 m) and one or two post-event images on
*different* grids. Preprocessing warps post-event image 1 onto the pre-event grid (bilinear) and
rasterises the annotations into:

* ``mask``      uint8 classes — 0 background, 1 building, 2 flooded building, 3 road, 4 flooded road
* ``buildings`` uint16 instance ids (1..N) for object-level evaluation, ``building_flooded`` flags
* ``road_center`` uint8 — 1-px road centrelines: 1 not flooded, 2 flooded (length-weighted eval)

Roads are buffered by ``lanes × 3.5 m / 2`` in UTM metres before rasterising.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.warp import reproject

CLASSES = ("background", "building", "flooded_building", "road", "flooded_road")
LANE_WIDTH_M = 3.5
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


@dataclass(frozen=True)
class SN8Tile:
    name: str  # e.g. "0_41_59"
    pre: str
    post: str

    @property
    def grid_xy(self) -> tuple[int, int]:
        _, x, y = self.name.split("_")
        return int(x), int(y)


def index_tiles(aoi_dir: Path) -> list[SN8Tile]:
    mapping = next(aoi_dir.glob("*_label_image_mapping.csv"))
    out = []
    with mapping.open(newline="") as f:
        for row in csv.DictReader(f):
            out.append(
                SN8Tile(
                    row["label"].removesuffix(".geojson"),
                    row["pre-event image"],
                    row["post-event image 1"],
                )
            )
    return sorted(out, key=lambda t: t.name)


def _is_flooded(v: Any) -> bool:
    return str(v).lower() in {"yes", "true", "1"}


def rasterize_labels(
    ann: gpd.GeoDataFrame, shape: tuple[int, int], transform, crs
) -> dict[str, np.ndarray]:
    mask = np.zeros(shape, np.uint8)
    inst = np.zeros(shape, np.uint16)
    road_center = np.zeros(shape, np.uint8)
    flooded_flags: list[bool] = []
    if len(ann) == 0:
        return {
            "mask": mask,
            "buildings": inst,
            "road_center": road_center,
            "building_flooded": np.zeros(0, bool),
        }
    ann = ann.to_crs(crs)
    is_bld = ann["building"].notna() & ann.geom_type.isin(["Polygon", "MultiPolygon"])
    is_road = ann["highway"].notna() & ann.geom_type.isin(["LineString", "MultiLineString"])
    roads = ann[is_road]
    if len(roads):
        utm = roads.estimate_utm_crs()
        lanes = roads["lanes"].fillna("2").astype(str).str.extract(r"(\d+)")[0].astype(float)
        lanes = lanes.fillna(2).clip(1, 6)
        buffered = roads.to_crs(utm).buffer(lanes.values * LANE_WIDTH_M / 2).to_crs(crs)
        flooded = roads["flooded"].map(_is_flooded).values
        for geoms, val in ((buffered[~flooded], 3), (buffered[flooded], 4)):
            if len(geoms):
                rasterize(((g, val) for g in geoms), out=mask, transform=transform)
        for geoms, val in ((roads.geometry[~flooded], 1), (roads.geometry[flooded], 2)):
            if len(geoms):
                rasterize(
                    ((g, val) for g in geoms),
                    out=road_center,
                    transform=transform,
                    all_touched=True,
                )
    blds = ann[is_bld]
    for i, (geom, fl) in enumerate(zip(blds.geometry, blds["flooded"], strict=True), start=1):
        f = _is_flooded(fl)
        flooded_flags.append(f)
        rasterize([(geom, i)], out=inst, transform=transform)
    if len(blds):
        fl_arr = np.array([False, *flooded_flags])
        has = inst > 0
        mask[has] = np.where(fl_arr[inst[has]], 2, 1)
    return {
        "mask": mask,
        "buildings": inst,
        "road_center": road_center,
        "building_flooded": np.array(flooded_flags, bool),
    }


def prepare_tile(tile: SN8Tile, aoi_dir: Path, out_dir: Path) -> dict[str, Any]:
    with rasterio.open(aoi_dir / "PRE-event" / tile.pre) as pre_src:
        pre = pre_src.read()
        shape, transform, crs = (pre_src.height, pre_src.width), pre_src.transform, pre_src.crs
    with rasterio.open(aoi_dir / "POST-event" / tile.post) as post_src:
        post = np.zeros((3, *shape), np.uint8)
        for b in range(3):
            reproject(
                rasterio.band(post_src, b + 1),
                post[b],
                dst_transform=transform,
                dst_crs=crs,
                resampling=Resampling.bilinear,
            )
    ann = gpd.read_file(aoi_dir / "annotations" / f"{tile.name}.geojson")
    labels = rasterize_labels(ann, shape, transform, crs)
    np.savez_compressed(out_dir / f"{tile.name}.npz", pre=pre[:3], post=post, **labels)
    counts = np.bincount(labels["mask"].ravel(), minlength=len(CLASSES))
    return {
        "tile": tile.name,
        "class_pixels": counts.tolist(),
        "buildings": int(len(labels["building_flooded"])),
        "flooded_buildings": int(labels["building_flooded"].sum()),
        "post_valid_fraction": float((post.sum(0) > 0).mean()),
    }


def spatial_block_split(
    tiles: list[SN8Tile], test_x: tuple[int, int], val_x: tuple[int, int]
) -> dict[str, list[str]]:
    """Split by contiguous ranges of the tile-grid x index, so neighbouring tiles (which share
    buildings and roads along their edges) never straddle train and test."""
    out: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for t in tiles:
        x, _ = t.grid_xy
        key = (
            "test"
            if test_x[0] <= x <= test_x[1]
            else "val"
            if val_x[0] <= x <= val_x[1]
            else "train"
        )
        out[key].append(t.name)
    return out


class SN8Dataset:
    """Returns dict(image FloatTensor[6,H,W] (pre RGB, post RGB; ImageNet-normalised),
    mask LongTensor[H,W], tile str). ``crop`` → random crops (train); else full tile padded to a
    multiple of 32 (padding labelled 255 = ignore)."""

    def __init__(
        self,
        names: list[str],
        root: Path,
        crop: int | None = None,
        augment: bool = False,
        seed: int = 0,
        focus_classes: tuple[int, ...] = (),
        focus_prob: float = 0.0,
    ) -> None:
        self.names = list(names)
        self.root = Path(root)
        self.crop = crop
        self.augment = augment
        # Rare-class sampling: with probability ``focus_prob`` a crop is centred on a random
        # pixel of ``focus_classes`` (when the tile has one), so flooded assets are seen often.
        self.focus_classes = focus_classes
        self.focus_prob = focus_prob
        self.rng = np.random.default_rng(seed)
        self._cache: dict[str, dict[str, np.ndarray]] = {}

    def load(self, name: str) -> dict[str, np.ndarray]:
        if name not in self._cache:
            with np.load(self.root / f"{name}.npz") as z:
                self._cache[name] = {k: z[k] for k in z.files}
        return self._cache[name]

    def __len__(self) -> int:
        return len(self.names)

    @staticmethod
    def to_input(pre: np.ndarray, post: np.ndarray) -> np.ndarray:
        x = np.concatenate([pre, post]).astype(np.float32) / 255.0
        mean = np.tile(IMAGENET_MEAN, 2)[:, None, None]
        std = np.tile(IMAGENET_STD, 2)[:, None, None]
        return (x - mean) / std

    def __getitem__(self, idx: int) -> dict[str, Any]:
        import torch

        d = self.load(self.names[idx])
        x = self.to_input(d["pre"], d["post"])
        y = d["mask"].astype(np.int64)
        h, w = y.shape
        if self.crop:
            r = int(self.rng.integers(0, h - self.crop + 1))
            c = int(self.rng.integers(0, w - self.crop + 1))
            if self.focus_classes and self.rng.random() < self.focus_prob:
                rows, cols = np.nonzero(np.isin(y, self.focus_classes))
                if rows.size:
                    k = int(self.rng.integers(0, rows.size))
                    r = int(np.clip(rows[k] - self.crop // 2, 0, h - self.crop))
                    c = int(np.clip(cols[k] - self.crop // 2, 0, w - self.crop))
            x, y = (
                x[:, r : r + self.crop, c : c + self.crop],
                y[r : r + self.crop, c : c + self.crop],
            )
        else:
            ph, pw = (-h) % 32, (-w) % 32
            x = np.pad(x, ((0, 0), (0, ph), (0, pw)))
            y = np.pad(y, ((0, ph), (0, pw)), constant_values=255)
        if self.augment:
            if self.rng.random() < 0.5:
                x, y = x[:, :, ::-1], y[:, ::-1]
            if self.rng.random() < 0.5:
                x, y = x[:, ::-1], y[::-1]
            k = int(self.rng.integers(0, 4))
            x, y = np.rot90(x, k, axes=(1, 2)), np.rot90(y, k)
        return {
            "image": torch.from_numpy(np.ascontiguousarray(x)),
            "mask": torch.from_numpy(np.ascontiguousarray(y)),
            "tile": self.names[idx],
        }


def object_level_scores(
    pred: np.ndarray,
    buildings: np.ndarray,
    building_flooded: np.ndarray,
    road_center: np.ndarray,
    min_detect: float = 0.2,
) -> dict[str, Any]:
    """Building/road flood attribution from a predicted class map (same grid as labels).

    Buildings: detected if ≥ ``min_detect`` of the footprint is predicted building (class 1/2);
    predicted flooded if class-2 pixels outnumber class-1 pixels inside the footprint.
    Roads: along ground-truth centreline pixels — detected if predicted road (3/4), flooded if 4.
    Returns raw counts so they can be pooled across tiles."""
    out = {
        "bld_total": 0,
        "bld_detected": 0,
        "bld_tp": 0,
        "bld_fp": 0,
        "bld_fn": 0,
        "bld_tn": 0,
        "road_px": 0,
        "road_detected_px": 0,
        "road_tp": 0,
        "road_fp": 0,
        "road_fn": 0,
        "road_tn": 0,
    }
    n = len(building_flooded)
    if n:
        ids = buildings.ravel()
        p = pred.ravel()
        sel = ids > 0
        area = np.bincount(ids[sel], minlength=n + 1)
        n1 = np.bincount(ids[sel & (p == 1)], minlength=n + 1)
        n2 = np.bincount(ids[sel & (p == 2)], minlength=n + 1)
        for i in range(1, n + 1):
            if area[i] == 0:
                continue
            out["bld_total"] += 1
            if (n1[i] + n2[i]) / area[i] < min_detect:
                if building_flooded[i - 1]:
                    out["bld_fn"] += 1
                continue
            out["bld_detected"] += 1
            pf, gf = n2[i] > n1[i], bool(building_flooded[i - 1])
            out["bld_tp" if pf and gf else "bld_fp" if pf else "bld_fn" if gf else "bld_tn"] += 1
    rc = road_center > 0
    if rc.any():
        g = road_center[rc] == 2
        p = pred[rc]
        det = (p == 3) | (p == 4)
        pf = p == 4
        out["road_px"] = int(rc.sum())
        out["road_detected_px"] = int(det.sum())
        out["road_tp"] = int((pf & g).sum())
        out["road_fp"] = int((pf & ~g).sum())
        out["road_fn"] = int((~pf & g).sum())
        out["road_tn"] = int((~pf & ~g).sum())
    return out


def summarize_object_scores(c: dict[str, int]) -> dict[str, float]:
    def f1(tp: int, fp: int, fn: int) -> dict[str, float]:
        p = tp / (tp + fp) if tp + fp else float("nan")
        r = tp / (tp + fn) if tp + fn else float("nan")
        return {
            "precision": p,
            "recall": r,
            "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan"),
        }

    return {
        "buildings_total": c["bld_total"],
        "building_detection_rate": c["bld_detected"] / c["bld_total"] if c["bld_total"] else None,
        "flooded_building": f1(c["bld_tp"], c["bld_fp"], c["bld_fn"]),
        "road_centerline_detection_rate": c["road_detected_px"] / c["road_px"]
        if c["road_px"]
        else None,
        "flooded_road_length": f1(c["road_tp"], c["road_fp"], c["road_fn"]),
    }


def write_split(path: Path, split: dict[str, list[str]], meta: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(
            {"meta": meta, "splits": split, "counts": {k: len(v) for k, v in split.items()}},
            indent=1,
        )
    )
