import numpy as np
import pytest
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin

from geospatial.raster import (
    check_alignment,
    footprint,
    is_cog,
    pixel_area_m2,
    read_info,
    reproject_to_match,
    validate_raster,
    write_cog,
)
from tests.geospatial.conftest import TRANSFORM


def test_read_info(tif):
    info = read_info(tif())
    assert (info.count, info.height, info.width) == (2, 32, 32)
    assert info.crs == "EPSG:4326"
    assert info.dtypes == ("float32", "float32")


def test_validate_good_raster(tif):
    res = validate_raster(
        tif(),
        expected_count=2,
        expected_dtype="float32",
        expected_shape=(32, 32),
        expected_crs="EPSG:4326",
        value_range=(-50, 5),
    )
    assert res.ok, res.errors


def test_validate_detects_structural_problems(tif):
    res = validate_raster(
        tif(), expected_count=3, expected_shape=(64, 64), expected_crs="EPSG:3857"
    )
    assert len(res.errors) == 3


def test_validate_detects_missing_crs(tmp_path):
    p = tmp_path / "nocrs.tif"
    with rasterio.open(p, "w", driver="GTiff", width=4, height=4, count=1, dtype="uint8") as d:
        d.write(np.zeros((1, 4, 4), np.uint8))
    errs = validate_raster(p).errors
    assert "missing CRS" in errs and "missing geotransform" in errs


def test_validate_values_and_nodata(tif):
    data = np.full((1, 10, 10), -10, np.float32)
    data[0, :6] = np.nan
    data[0, 9, 9] = 99
    res = validate_raster(tif(data=data), value_range=(-50, 5), max_nodata_fraction=0.5)
    assert any("outside" in e for e in res.errors)
    assert any("nodata fraction 0.600" in w for w in res.warnings)


def test_validate_all_nodata_and_allowed_values(tif):
    assert (
        "all pixels are nodata"
        in validate_raster(
            tif(data=np.full((1, 4, 4), np.nan, np.float32)), max_nodata_fraction=1.0
        ).errors
    )
    lab = np.array([[[-1, 0], [1, 2]]], np.int16)
    assert any(
        "unexpected values [2]" in e
        for e in validate_raster(tif("l.tif", data=lab), allowed_values={-1, 0, 1}).errors
    )


def test_validate_unreadable_file(tmp_path):
    p = tmp_path / "junk.tif"
    p.write_bytes(b"not a tiff")
    assert not validate_raster(p).ok


def test_alignment(tif):
    a = tif("a.tif")
    assert check_alignment(a, tif("b.tif")) == []
    shifted = from_origin(-64.0, -14.0, 8.983e-05, 8.983e-05)
    assert check_alignment(a, tif("c.tif", transform=shifted)) == ["geotransform differs"]
    assert any("CRS" in p for p in check_alignment(a, tif("d.tif", crs="EPSG:32720")))


def test_reproject_to_match_aligns_grid(tif, tmp_path):
    ref = tif("ref.tif")
    coarse = from_origin(-65.0, -14.0, 2 * 8.983e-05, 2 * 8.983e-05)
    lab = np.zeros((1, 16, 16), np.uint8)
    lab[0, :8, :8] = 1
    src = tif("src.tif", data=lab, transform=coarse)
    out = reproject_to_match(src, ref, tmp_path / "out.tif", resampling=Resampling.nearest)
    assert check_alignment(out, ref) == []
    with rasterio.open(out) as d:
        arr = d.read(1)
    assert arr[:16, :16].all() and not arr[16:, 16:].any()  # block upsampled 2×
    with rasterio.open(out) as d:
        assert d.nodata is None  # 0 is a real class, must not become nodata


def test_reproject_outside_footprint_uses_nodata(tif, tmp_path):
    ref = tif("ref.tif")
    half = np.ones((1, 32, 16), np.uint8)
    src = tif("half.tif", data=half)  # covers only the left half of ref
    out = reproject_to_match(src, ref, tmp_path / "o.tif", dst_nodata=255)
    with rasterio.open(out) as d:
        arr = d.read(1)
    assert (arr[:, :16] == 1).all() and (arr[:, 16:] == 255).all()


def test_write_cog_roundtrip(tmp_path):
    arr = np.random.default_rng(1).integers(0, 2, (1, 1024, 1024), dtype=np.uint8)
    p = write_cog(
        arr,
        tmp_path / "m.tif",
        crs="EPSG:4326",
        transform=TRANSFORM,
        nodata=255,
        descriptions=["flood"],
    )
    assert is_cog(p)
    with rasterio.open(p) as d:
        assert np.array_equal(d.read(), arr)
        assert d.descriptions == ("flood",) and d.nodata == 255


def test_footprint_and_pixel_area(tif):
    p = tif()
    geom = footprint(p)
    assert geom["type"] == "Polygon"
    xs = [c[0] for c in geom["coordinates"][0]]
    assert min(xs) == pytest.approx(-65.0)
    # 8.983e-5 deg ≈ 10 m at the equator; at 14°S, ~9.7 m × 9.9 m
    assert 90 < pixel_area_m2(p) < 100


def test_pixel_area_projected(tif):
    p = tif(crs="EPSG:32720", transform=from_origin(500000, 8400000, 10, 10))
    assert pixel_area_m2(p) == pytest.approx(100.0)
