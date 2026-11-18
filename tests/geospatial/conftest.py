from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

TRANSFORM = from_origin(-65.0, -14.0, 8.983e-05, 8.983e-05)


def write_tif(path: Path, data: np.ndarray, *, crs="EPSG:4326", transform=TRANSFORM, nodata=None):
    if data.ndim == 2:
        data = data[None]
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=data.shape[2],
        height=data.shape[1],
        count=data.shape[0],
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data)
    return path


@pytest.fixture
def tif(tmp_path):
    def _make(name="a.tif", data=None, **kw):
        if data is None:
            data = np.random.default_rng(0).normal(-15, 3, (2, 32, 32)).astype(np.float32)
        return write_tif(tmp_path / name, data, **kw)

    return _make
