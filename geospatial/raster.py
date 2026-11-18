"""Raster I/O, validation and alignment utilities (rasterio / GDAL)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject, transform_geom
from shapely.geometry import box, mapping, shape

PathLike = str | Path


@dataclass(frozen=True)
class RasterInfo:
    path: str
    width: int
    height: int
    count: int
    dtypes: tuple[str, ...]
    crs: str | None
    transform: tuple[float, ...]
    res: tuple[float, float]
    bounds: tuple[float, float, float, float]
    nodata: float | None
    descriptions: tuple[str | None, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_info(path: PathLike) -> RasterInfo:
    with rasterio.open(path) as src:
        return RasterInfo(
            path=str(path),
            width=src.width,
            height=src.height,
            count=src.count,
            dtypes=tuple(src.dtypes),
            crs=src.crs.to_string() if src.crs else None,
            transform=tuple(src.transform)[:6],
            res=src.res,
            bounds=tuple(src.bounds),
            nodata=src.nodata,
            descriptions=tuple(src.descriptions),
        )


@dataclass
class ValidationResult:
    path: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_raster(
    path: PathLike,
    *,
    expected_count: int | None = None,
    expected_dtype: str | None = None,
    expected_shape: tuple[int, int] | None = None,
    expected_crs: str | None = None,
    value_range: tuple[float, float] | None = None,
    allowed_values: set[int] | None = None,
    max_nodata_fraction: float | None = None,
) -> ValidationResult:
    """Check a raster's structure and values. Never raises for bad data; returns errors instead."""
    res = ValidationResult(str(path))
    try:
        src = rasterio.open(path)
    except rasterio.errors.RasterioIOError as exc:
        res.errors.append(f"unreadable: {exc}")
        return res
    with src:
        if src.crs is None:
            res.errors.append("missing CRS")
        elif expected_crs and CRS.from_user_input(expected_crs) != src.crs:
            res.errors.append(f"CRS {src.crs} != {expected_crs}")
        if src.transform == Affine.identity():
            res.errors.append("missing geotransform")
        if expected_count is not None and src.count != expected_count:
            res.errors.append(f"band count {src.count} != {expected_count}")
        if expected_dtype and any(d != expected_dtype for d in src.dtypes):
            res.errors.append(f"dtype {src.dtypes} != {expected_dtype}")
        if expected_shape and (src.height, src.width) != expected_shape:
            res.errors.append(f"shape {(src.height, src.width)} != {expected_shape}")
        if value_range is None and allowed_values is None and max_nodata_fraction is None:
            return res

        data = src.read(masked=False)
        invalid = ~np.isfinite(data) if data.dtype.kind == "f" else np.zeros(data.shape, bool)
        if src.nodata is not None and not np.isnan(src.nodata):
            invalid |= data == src.nodata
        frac = float(invalid.any(axis=0).mean())
        if max_nodata_fraction is not None and frac > max_nodata_fraction:
            res.warnings.append(f"nodata fraction {frac:.3f} > {max_nodata_fraction}")
        if frac == 1.0:
            res.errors.append("all pixels are nodata")
            return res
        valid = data[~invalid]
        if value_range is not None and valid.size:
            lo, hi = float(valid.min()), float(valid.max())
            if lo < value_range[0] or hi > value_range[1]:
                res.errors.append(f"values [{lo:.3f}, {hi:.3f}] outside {value_range}")
        if allowed_values is not None:
            extra = set(np.unique(data).tolist()) - allowed_values
            if extra:
                res.errors.append(f"unexpected values {sorted(extra)[:10]}")
    return res


def check_alignment(a: PathLike, b: PathLike, *, atol: float = 1e-9) -> list[str]:
    """Return a list of differences that would break pixel-wise pairing of two rasters."""
    ia, ib = read_info(a), read_info(b)
    problems = []
    if ia.crs != ib.crs:
        problems.append(f"CRS differs: {ia.crs} vs {ib.crs}")
    if (ia.height, ia.width) != (ib.height, ib.width):
        problems.append(f"shape differs: {(ia.height, ia.width)} vs {(ib.height, ib.width)}")
    if not np.allclose(ia.transform, ib.transform, atol=atol, rtol=0):
        problems.append("geotransform differs")
    return problems


def reproject_to_match(
    src_path: PathLike,
    ref_path: PathLike,
    out_path: PathLike,
    *,
    resampling: Resampling = Resampling.nearest,
    dst_nodata: float | None = None,
) -> Path:
    """Warp ``src`` onto ``ref``'s grid (CRS, transform, shape). Use nearest for labels,
    bilinear for continuous imagery."""
    out_path = Path(out_path)
    with rasterio.open(ref_path) as ref, rasterio.open(src_path) as src:
        profile = src.profile.copy()
        profile.update(
            crs=ref.crs, transform=ref.transform, width=ref.width, height=ref.height, driver="GTiff"
        )
        is_float = np.dtype(src.dtypes[0]).kind == "f"
        if dst_nodata is None:
            dst_nodata = src.nodata if src.nodata is not None else (np.nan if is_float else None)
        # Never invent nodata=0 for integer rasters: 0 is usually a real class ("not water").
        # Without a nodata value, pixels outside the source footprint are filled with 0.
        fill = 0 if dst_nodata is None else dst_nodata
        out = np.full((src.count, ref.height, ref.width), fill, dtype=src.dtypes[0])
        reproject(
            source=src.read(),
            destination=out,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=ref.transform,
            dst_crs=ref.crs,
            dst_nodata=dst_nodata,
            resampling=resampling,
        )
        profile["nodata"] = dst_nodata
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(out)
            dst.descriptions = src.descriptions
    return out_path


def write_cog(
    array: np.ndarray,
    path: PathLike,
    *,
    crs: CRS | str,
    transform: Affine,
    nodata: float | None = None,
    descriptions: list[str] | None = None,
    compress: str = "DEFLATE",
    blocksize: int = 512,
) -> Path:
    """Write a (bands, H, W) or (H, W) array as a Cloud Optimized GeoTIFF."""
    if array.ndim == 2:
        array = array[None]
    path = Path(path)
    with rasterio.open(
        path,
        "w",
        driver="COG",
        width=array.shape[2],
        height=array.shape[1],
        count=array.shape[0],
        dtype=array.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress=compress,
        blocksize=blocksize,
        overviews="AUTO",
    ) as dst:
        dst.write(array)
        if descriptions:
            for i, d in enumerate(descriptions, 1):
                dst.set_band_description(i, d)
    return path


def is_cog(path: PathLike) -> bool:
    with rasterio.open(path) as src:
        return src.tags(ns="IMAGE_STRUCTURE").get("LAYOUT") == "COG" or (
            src.profile.get("tiled", False) and bool(src.overviews(1))
        )


def footprint(path: PathLike, dst_crs: str = "EPSG:4326") -> dict[str, Any]:
    """Bounding-box footprint as a GeoJSON geometry in ``dst_crs``."""
    with rasterio.open(path) as src:
        geom = mapping(box(*src.bounds))
        if src.crs and CRS.from_user_input(dst_crs) != src.crs:
            geom = transform_geom(src.crs, dst_crs, geom)
    return dict(shape(geom).__geo_interface__)


def pixel_area_m2(path: PathLike) -> float:
    """Approximate area of one pixel in m² at the raster centre (handles geographic CRSs)."""
    from pyproj import Geod

    with rasterio.open(path) as src:
        if src.crs and src.crs.is_projected:
            return abs(src.transform.a * src.transform.e) * src.crs.linear_units_factor[1] ** 2
        cx, cy = src.xy(src.height // 2, src.width // 2)
        dx, dy = src.res
    poly = box(cx, cy, cx + dx, cy + dy)
    area, _ = Geod(ellps="WGS84").geometry_area_perimeter(poly)
    return abs(area)
