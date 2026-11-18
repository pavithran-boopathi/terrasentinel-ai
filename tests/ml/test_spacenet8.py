import geopandas as gpd
import numpy as np
import pytest
from rasterio.transform import from_origin
from shapely.geometry import LineString, box

from ml.data.spacenet8 import (
    SN8Dataset,
    SN8Tile,
    object_level_scores,
    rasterize_labels,
    spatial_block_split,
    summarize_object_scores,
)

RES = 4e-6  # ~0.3-0.45 m at 50.5°N
TF = from_origin(7.0, 50.5, RES, RES)


def ann(rows):
    return gpd.GeoDataFrame(rows, geometry=[r.pop("geometry") for r in rows], crs="EPSG:4326")


def px_box(c0, r0, c1, r1):
    return box(7.0 + c0 * RES, 50.5 - r1 * RES, 7.0 + c1 * RES, 50.5 - r0 * RES)


def test_rasterize_classes_instances_and_centerlines():
    rows = [
        {
            "building": "yes",
            "highway": None,
            "lanes": None,
            "flooded": "yes",
            "geometry": px_box(10, 10, 30, 30),
        },
        {
            "building": "yes",
            "highway": None,
            "lanes": None,
            "flooded": None,
            "geometry": px_box(50, 10, 70, 30),
        },
        {
            "building": None,
            "highway": "residential",
            "lanes": "2",
            "flooded": "yes",
            "geometry": LineString(
                [(7.0 + 5 * RES, 50.5 - 80 * RES), (7.0 + 95 * RES, 50.5 - 80 * RES)]
            ),
        },
    ]
    out = rasterize_labels(ann(rows), (100, 100), TF, "EPSG:4326")
    m = out["mask"]
    assert m[20, 20] == 2 and m[20, 60] == 1 and m[80, 50] == 4 and m[0, 0] == 0
    # 2 lanes × 3.5 m → 7 m wide road ≈ 7 m / (RES·111 km) ≈ 16 px tall
    assert 10 < (m[:, 50] == 4).sum() < 22
    assert out["building_flooded"].tolist() == [True, False]
    assert set(np.unique(out["buildings"])) == {0, 1, 2}
    assert (out["road_center"] == 2).sum() >= 85


def test_rasterize_empty_annotations():
    out = rasterize_labels(gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"), (8, 8), TF, "EPSG:4326")
    assert out["mask"].sum() == 0 and len(out["building_flooded"]) == 0


def test_spatial_block_split_has_no_shared_columns():
    tiles = [SN8Tile(f"0_{x}_{y}", "a", "b") for x in range(10, 30) for y in range(3)]
    s = spatial_block_split(tiles, test_x=(25, 29), val_x=(20, 24))
    xs = {k: {int(n.split("_")[1]) for n in v} for k, v in s.items()}
    assert not (xs["train"] & xs["test"]) and not (xs["train"] & xs["val"])
    assert sum(len(v) for v in s.values()) == len(tiles)


def test_object_level_scores():
    bld = np.zeros((10, 10), np.uint16)
    bld[0:4, 0:4] = 1  # GT flooded
    bld[0:4, 6:10] = 2  # GT dry
    bld[6:10, 0:4] = 3  # GT flooded, missed
    pred = np.zeros((10, 10), np.uint8)
    pred[0:4, 0:4] = 2
    pred[0:4, 6:10] = 2  # wrongly flooded
    rc = np.zeros((10, 10), np.uint8)
    rc[8, :] = 2
    pred[8, :5] = 4
    pred[8, 5:] = 3
    c = object_level_scores(pred, bld, np.array([True, False, True]), rc)
    assert (c["bld_tp"], c["bld_fp"], c["bld_fn"], c["bld_detected"]) == (1, 1, 1, 2)
    assert (c["road_tp"], c["road_fn"], c["road_detected_px"]) == (5, 5, 10)
    s = summarize_object_scores(c)
    assert s["flooded_building"]["f1"] == pytest.approx(0.5)
    assert s["flooded_road_length"]["recall"] == pytest.approx(0.5)


def test_dataset_crops_and_pads(tmp_path):
    rng = np.random.default_rng(0)
    np.savez_compressed(
        tmp_path / "0_1_2.npz",
        pre=rng.integers(0, 255, (3, 70, 90), np.uint8),
        post=rng.integers(0, 255, (3, 70, 90), np.uint8),
        mask=rng.integers(0, 5, (70, 90), np.uint8),
    )
    full = SN8Dataset(["0_1_2"], tmp_path)[0]
    assert full["image"].shape == (6, 96, 96) and (full["mask"][70:] == 255).all()
    crop = SN8Dataset(["0_1_2"], tmp_path, crop=32, augment=True)[0]
    assert crop["image"].shape == (6, 32, 32) and crop["mask"].max() <= 4
