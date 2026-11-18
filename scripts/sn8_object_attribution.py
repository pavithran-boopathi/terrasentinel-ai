#!/usr/bin/env python3
"""Object-level flood attribution on SpaceNet 8 given *known* asset geometry.

    python scripts/sn8_object_attribution.py

Realistic setting for exposure analysis: building footprints / road centrelines come from a map
(e.g. OpenStreetMap); the question is which of them are flooded. For every building (footprint +
a 3 m surrounding ring) and road segment (buffered centreline), features are statistics of the
pre- and post-event RGB and of their difference; a gradient-boosted classifier is trained on the
train block and evaluated on the spatially held-out test block. Compared with the end-to-end
deep model and trivial baselines. Writes docs/results/sn8_object_attribution.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
from rasterio.features import rasterize
from rasterio.transform import Affine

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ml.data.spacenet8 import LANE_WIDTH_M, _is_flooded, index_tiles  # noqa: E402

AOI = REPO / "data/raw/spacenet8/Germany_Training_Public"
PROC = REPO / "data/processed/spacenet8"
RING_M = 3.0


def stats(px: np.ndarray) -> list[float]:
    """px: (C, N) → per-channel mean, std, p10, p90."""
    if px.shape[1] == 0:
        return [np.nan] * (px.shape[0] * 4)
    return [*px.mean(1), *px.std(1), *np.percentile(px, 10, 1), *np.percentile(px, 90, 1)]


def tile_features(name: str, pre_file: str):
    import rasterio

    d = np.load(PROC / "tiles" / f"{name}.npz")
    pre, post = d["pre"].astype(np.float32) / 255, d["post"].astype(np.float32) / 255
    with rasterio.open(AOI / "PRE-event" / pre_file) as src:
        transform: Affine = src.transform
        crs = src.crs
    post_valid = post.sum(0) > 0
    diff = post - pre
    hsv_like = np.stack([post[0] - post[1], post[2] - post[1], post.mean(0)])  # brownness/blueness
    stack = np.concatenate([pre, post, diff, hsv_like])  # 12 channels
    ann = gpd.read_file(AOI / "annotations" / f"{name}.geojson")
    if not len(ann):
        return []
    ann = ann.to_crs(crs)
    utm = ann.estimate_utm_crs()
    rows = []
    shape = pre.shape[1:]
    for _, f in ann.iterrows():
        g = f.geometry
        if g is None or g.is_empty:
            continue
        is_b = f.get("building") is not None and g.geom_type in ("Polygon", "MultiPolygon")
        is_r = f.get("highway") is not None and g.geom_type in ("LineString", "MultiLineString")
        if not (is_b or is_r):
            continue
        gs = gpd.GeoSeries([g], crs=crs).to_crs(utm)
        if is_b:
            inner, outer = gs.iloc[0], gs.buffer(RING_M).iloc[0].difference(gs.iloc[0])
        else:
            lanes = float(str(f.get("lanes") or "2").split(";")[0] or 2)
            inner = gs.buffer(max(1.0, min(lanes, 6)) * LANE_WIDTH_M / 2).iloc[0]
            outer = (
                gs.buffer(max(1.0, min(lanes, 6)) * LANE_WIDTH_M / 2 + RING_M)
                .iloc[0]
                .difference(inner)
            )
        back = gpd.GeoSeries([inner, outer], crs=utm).to_crs(crs)
        m_in = (
            rasterize(
                [(back.iloc[0], 1)], out_shape=shape, transform=transform, all_touched=True
            ).astype(bool)
            & post_valid
        )
        m_out = (
            rasterize(
                [(back.iloc[1], 1)], out_shape=shape, transform=transform, all_touched=True
            ).astype(bool)
            & post_valid
        )
        if m_in.sum() < 4:
            continue
        feats = stats(stack[:, m_in]) + stats(stack[:, m_out])
        rows.append(
            {
                "tile": name,
                "kind": "building" if is_b else "road",
                "flooded": _is_flooded(f.get("flooded")),
                "weight": float(gs.length.iloc[0]) if is_r else 1.0,
                "x": feats,
            }
        )
    return rows


def f1(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> dict[str, float]:
    tp, fp, fn = (w * (p & y)).sum(), (w * (p & ~y)).sum(), (w * (~p & y)).sum()
    return {
        "precision": float(tp / (tp + fp)) if tp + fp else float("nan"),
        "recall": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "f1": float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else float("nan"),
    }


def main() -> None:
    from concurrent.futures import ProcessPoolExecutor

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    split = json.loads((PROC / "split.json").read_text())["splits"]
    which = {n: k for k, ns in split.items() for n in ns}
    tiles = index_tiles(AOI)
    with ProcessPoolExecutor(6) as ex:
        per_tile = list(ex.map(tile_features, [t.name for t in tiles], [t.pre for t in tiles]))
    rows = [r for rs in per_tile for r in rs]
    out: dict = {"ring_m": RING_M, "features": 12 * 4 * 2, "results": {}}
    for kind in ("building", "road"):
        data = {
            s: [r for r in rows if r["kind"] == kind and which[r["tile"]] == s]
            for s in ("train", "val", "test")
        }
        X = {s: np.array([r["x"] for r in v], float) for s, v in data.items()}
        y = {s: np.array([r["flooded"] for r in v], bool) for s, v in data.items()}
        w = {s: np.array([r["weight"] for r in v], float) for s, v in data.items()}
        clf = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=0,
        )
        clf.fit(np.concatenate([X["train"]]), y["train"], sample_weight=w["train"])
        # threshold chosen on validation (spatial val block), applied to test
        pv = clf.predict_proba(X["val"])[:, 1]
        grid = np.linspace(0.05, 0.95, 19)
        thr = float(grid[np.argmax([f1(y["val"], pv > t, w["val"])["f1"] for t in grid])])
        pt = clf.predict_proba(X["test"])[:, 1]
        out["results"][kind] = {
            "n": {s: int(len(v)) for s, v in y.items()},
            "flooded_fraction_test": float((w["test"] * y["test"]).sum() / w["test"].sum()),
            "val_threshold": thr,
            "gbt_test": f1(y["test"], pt > thr, w["test"]),
            "gbt_test_at_0.5": f1(y["test"], pt > 0.5, w["test"]),
            "all_flooded_test": f1(y["test"], np.ones_like(y["test"]), w["test"]),
            # threshold-free: 0.5 = no signal
            "gbt_test_auroc": float(roc_auc_score(y["test"], pt, sample_weight=w["test"])),
            "gbt_val_auroc": float(roc_auc_score(y["val"], pv, sample_weight=w["val"])),
        }
        print(kind, json.dumps(out["results"][kind], indent=None, default=float))
    dest = REPO / "docs/results/sn8_object_attribution.json"
    dest.write_text(json.dumps(out, indent=1, default=float))
    print(f"wrote {dest.relative_to(REPO)}")


if __name__ == "__main__":
    main()
