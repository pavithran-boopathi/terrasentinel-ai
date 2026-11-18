"""Small WGS84 PNG previews of rasters for web maps (MapLibre ``image`` sources).

Previews are warped to EPSG:4326 so their four corners can be placed by longitude/latitude.
Sizes are capped (default 1024 px on the long side), so this is cheap even for large scenes.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

from geospatial.viz import sar_false_color


def _read_wgs84(path: str | Path, max_size: int, resampling: Resampling):
    with rasterio.open(path) as src, WarpedVRT(src, crs="EPSG:4326", resampling=resampling) as vrt:
        scale = max(vrt.width, vrt.height) / max_size
        h, w = max(1, round(vrt.height / max(scale, 1))), max(1, round(vrt.width / max(scale, 1)))
        data = vrt.read(out_shape=(vrt.count, h, w), resampling=resampling, masked=True)
        return data, tuple(vrt.bounds)


def _png(rgba: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, "PNG", optimize=True)
    return buf.getvalue()


def sar_preview(path: str | Path, max_size: int = 1024) -> tuple[bytes, tuple[float, ...]]:
    data, bounds = _read_wgs84(path, max_size, Resampling.average)
    arr = data.filled(np.nan).astype(np.float32)
    rgb = sar_false_color(arr[0], arr[1])
    alpha = np.where(np.isfinite(arr).all(0), 255, 0).astype(np.uint8)
    return _png(np.dstack([rgb, alpha])), bounds


def mask_preview(
    path: str | Path, max_size: int = 1024, color: tuple[int, int, int, int] = (30, 144, 255, 200)
) -> tuple[bytes, tuple]:
    data, bounds = _read_wgs84(path, max_size, Resampling.nearest)
    m = data.filled(255)[0]
    rgba = np.zeros((*m.shape, 4), np.uint8)
    rgba[m == 1] = color
    return _png(rgba), bounds
