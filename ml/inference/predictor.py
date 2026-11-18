"""Flood segmentation of a Sentinel-1 GeoTIFF of any size → georeferenced outputs.

Input contract: GeoTIFF with 2 bands, VV then VH, backscatter in **dB** (as in Sen1Floods11),
nodata as NaN or the file's nodata value. Outputs (all in the input CRS/grid):

* ``probability.tif`` — COG uint8, P(water) × 250, 255 = nodata
* ``flood_mask.tif``  — COG uint8, 1 water / 0 not water, 255 = nodata
* ``flood_polygons.geojson`` — water polygons in EPSG:4326 with geodesic ``area_m2``
* summary dict — flooded / valid area (km²), pixel counts, timings, model id
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
from pyproj import Geod
from rasterio.features import shapes
from rasterio.warp import transform_geom
from rasterio.windows import Window
from shapely.geometry import mapping, shape

from geospatial.sentinel1 import InputValidationError, validate_input  # noqa: F401
from ml.data.sen1floods11 import NormStats, normalize
from ml.inference.tiling import TileSpec, tiled_predict
from ml.models.factory import model_from_config
from ml.training.engine import select_device

PROB_SCALE = 250
NODATA_U8 = 255
_GEOD = Geod(ellps="WGS84")


def row_pixel_areas_m2(src: rasterio.io.DatasetReader) -> np.ndarray:
    """Geodesic area of one pixel for every row (varies with latitude for geographic CRSs)."""
    t = src.transform
    if src.crs.is_projected:
        a = abs(t.a * t.e) * src.crs.linear_units_factor[1] ** 2
        return np.full(src.height, a, np.float64)
    out = np.empty(src.height, np.float64)
    x0 = t.c
    for r in range(src.height):
        y0 = t.f + r * t.e
        lons = [x0, x0 + t.a, x0 + t.a, x0]
        lats = [y0, y0, y0 + t.e, y0 + t.e]
        out[r] = abs(_GEOD.polygon_area_perimeter(lons, lats)[0])
    return out


@dataclass
class FloodPredictor:
    model: torch.nn.Module
    stats: NormStats
    device: torch.device
    threshold: float
    model_id: str

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str = "auto",
        threshold: float | None = None,
        model_id: str | None = None,
    ) -> FloodPredictor:
        state = torch.load(path, map_location="cpu", weights_only=False)
        cfg = state["config"]
        model = model_from_config(cfg)
        model.load_state_dict(state["model"])
        dev = select_device(device)
        model.to(dev).eval()
        stats = NormStats(**state["meta"]["norm_stats"])
        thr = threshold if threshold is not None else float(cfg.get("threshold", 0.5))
        return cls(model, stats, dev, thr, model_id or state["meta"]["run_name"])

    @torch.no_grad()
    def predict_batch(self, x: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(x).to(self.device)
        return torch.sigmoid(self.model(t)[:, 0]).float().cpu().numpy()

    def run(
        self,
        src_path: str | Path,
        out_dir: str | Path,
        *,
        tile: int = 512,
        overlap: int = 64,
        batch: int = 4,
        min_polygon_px: int = 4,
        progress: Callable[[float], None] | None = None,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        info = validate_input(src_path)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with rasterio.open(src_path) as src:
            nod = src.nodata

            def read_tile(s: TileSpec, size: int) -> np.ndarray:
                win = Window(s.col, s.row, size, size)
                arr = src.read(window=win, boundless=True, fill_value=np.nan).astype(np.float32)
                if nod is not None and not np.isnan(nod):
                    arr[arr == nod] = np.nan
                return normalize(arr, self.stats)

            probs = tiled_predict(
                read_tile,
                self.predict_batch,
                src.height,
                src.width,
                tile=tile,
                overlap=overlap,
                batch=batch,
                workdir=out_dir / ".tiles",
                progress=progress,
            )
            # Validity: every band finite and not nodata (read in row strips to bound memory).
            valid = np.zeros((src.height, src.width), bool)
            step = 1024
            for r in range(0, src.height, step):
                h = min(step, src.height - r)
                a = src.read(window=Window(0, r, src.width, h)).astype(np.float32)
                ok = np.isfinite(a).all(0)
                if nod is not None and not np.isnan(nod):
                    ok &= (a != nod).all(0)
                valid[r : r + h] = ok
            profile = {
                "driver": "COG",
                "width": src.width,
                "height": src.height,
                "count": 1,
                "dtype": "uint8",
                "crs": src.crs,
                "transform": src.transform,
                "nodata": NODATA_U8,
                "compress": "DEFLATE",
                "blocksize": 512,
                "overviews": "AUTO",
            }
            prob_u8 = np.where(valid, np.round(np.asarray(probs) * PROB_SCALE), NODATA_U8)
            prob_u8 = prob_u8.astype(np.uint8)
            mask = np.where(
                valid, (np.asarray(probs) > self.threshold).astype(np.uint8), NODATA_U8
            ).astype(np.uint8)
            with rasterio.open(out_dir / "probability.tif", "w", **profile) as dst:
                dst.write(prob_u8, 1)
                dst.update_tags(scale=f"1/{PROB_SCALE}", model=self.model_id)
            with rasterio.open(out_dir / "flood_mask.tif", "w", **profile) as dst:
                dst.write(mask, 1)
                dst.update_tags(threshold=str(self.threshold), model=self.model_id)

            row_area = row_pixel_areas_m2(src)
            water = mask == 1
            flooded_m2 = float((water.sum(axis=1) * row_area).sum())
            valid_m2 = float((valid.sum(axis=1) * row_area).sum())

            features = []
            min_area = min_polygon_px * float(row_area.mean())
            for geom, val in shapes(mask, mask=water, transform=src.transform, connectivity=8):
                if val != 1:
                    continue
                g4326 = (
                    transform_geom(src.crs, "EPSG:4326", geom)
                    if src.crs.to_epsg() != 4326
                    else geom
                )
                area = abs(_GEOD.geometry_area_perimeter(shape(g4326))[0])
                if area < min_area:
                    continue
                features.append(
                    {
                        "type": "Feature",
                        "geometry": mapping(shape(g4326)),
                        "properties": {"class": "predicted_flood_water", "area_m2": round(area, 1)},
                    }
                )
        del probs
        for f in (out_dir / ".tiles").glob("*"):
            f.unlink()
        (out_dir / ".tiles").rmdir()
        (out_dir / "flood_polygons.geojson").write_text(
            json.dumps(
                {
                    "type": "FeatureCollection",
                    "features": features,
                    "properties": {
                        "model": self.model_id,
                        "threshold": self.threshold,
                        "note": "Predicted open flood water from SAR; not a damage map.",
                    },
                }
            )
        )
        summary = {
            "model": self.model_id,
            "threshold": self.threshold,
            "device": self.device.type,
            "input": info,
            "pixels_valid": int(valid.sum()),
            "pixels_flooded": int(water.sum()),
            "flooded_area_km2": flooded_m2 / 1e6,
            "valid_area_km2": valid_m2 / 1e6,
            "flooded_fraction_of_valid": flooded_m2 / valid_m2 if valid_m2 else None,
            "polygons": len(features),
            "seconds": round(time.perf_counter() - t0, 2),
            "tile": tile,
            "overlap": overlap,
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
        return summary
