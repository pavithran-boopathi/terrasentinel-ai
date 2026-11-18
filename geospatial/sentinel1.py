"""Sentinel-1 input contract checks (no ML dependencies, safe to use in the API process)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio


class InputValidationError(ValueError):
    """The raster does not satisfy the input contract."""


def validate_input(path: str | Path, max_pixels: int = 400_000_000) -> dict[str, Any]:
    try:
        src = rasterio.open(path)
    except rasterio.errors.RasterioIOError as exc:
        raise InputValidationError(f"not a readable raster: {exc}") from None
    with src:
        if src.count != 2:
            raise InputValidationError(f"expected 2 bands (VV, VH in dB), got {src.count}")
        if src.crs is None:
            raise InputValidationError("raster has no CRS")
        if np.dtype(src.dtypes[0]).kind != "f":
            raise InputValidationError(f"expected float dB backscatter, got {src.dtypes[0]}")
        if src.width * src.height > max_pixels:
            raise InputValidationError(f"raster too large: {src.width}×{src.height}")
        desc = [d.upper() if d else None for d in src.descriptions]
        if all(desc) and desc != ["VV", "VH"]:
            raise InputValidationError(f"band descriptions {desc} != ['VV', 'VH']")
        return {
            "width": src.width,
            "height": src.height,
            "crs": src.crs.to_string(),
            "bounds": tuple(src.bounds),
            "res": src.res,
        }
