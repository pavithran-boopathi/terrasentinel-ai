#!/usr/bin/env python3
"""Preprocess SpaceNet 8 tiles (post→pre grid warp, label rasters) and write a spatial split.

python scripts/prepare_spacenet8.py [--aoi Germany_Training_Public]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ml.data.spacenet8 import (  # noqa: E402
    CLASSES,
    index_tiles,
    prepare_tile,
    spatial_block_split,
    write_split,
)

# Chosen after inspecting per-column tile and flooded-building counts (printed below) so that
# val and test are contiguous blocks that each contain flooded buildings and roads.
TEST_X = (39, 45)
VAL_X = (34, 38)


def _work(args):
    tile, aoi, out = args
    return prepare_tile(tile, aoi, out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aoi", default="Germany_Training_Public")
    ap.add_argument("--raw", type=Path, default=REPO / "data/raw/spacenet8")
    ap.add_argument("--out", type=Path, default=REPO / "data/processed/spacenet8")
    a = ap.parse_args()
    aoi = a.raw / a.aoi
    tiles_dir = a.out / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    tiles = index_tiles(aoi)
    with ProcessPoolExecutor(6) as ex:
        stats = list(ex.map(_work, [(t, aoi, tiles_dir) for t in tiles]))
    by = {s["tile"]: s for s in stats}
    col: Counter = Counter()
    col_fl: Counter = Counter()
    for t in tiles:
        x, _ = t.grid_xy
        col[x] += 1
        col_fl[x] += by[t.name]["flooded_buildings"]
    print("x: tiles / flooded buildings")
    print(" ".join(f"{x}:{col[x]}/{col_fl[x]}" for x in sorted(col)))
    split = spatial_block_split(tiles, TEST_X, VAL_X)
    summary = {}
    for k, names in split.items():
        px = [sum(by[n]["class_pixels"][i] for n in names) for i in range(len(CLASSES))]
        summary[k] = {
            "tiles": len(names),
            "buildings": sum(by[n]["buildings"] for n in names),
            "flooded_buildings": sum(by[n]["flooded_buildings"] for n in names),
            "class_pixel_fraction": {
                c: round(p / max(1, sum(px)), 5) for c, p in zip(CLASSES, px, strict=True)
            },
        }
    write_split(
        a.out / "split.json",
        split,
        {"aoi": a.aoi, "test_x": TEST_X, "val_x": VAL_X, "summary": summary},
    )
    (a.out / "tile_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
