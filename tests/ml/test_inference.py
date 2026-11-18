import json

import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_origin
from torch import nn

from ml.data.sen1floods11 import NormStats
from ml.inference.predictor import (
    NODATA_U8,
    FloodPredictor,
    InputValidationError,
    validate_input,
)
from ml.inference.tiling import hann_window, tile_grid, tiled_predict
from ml.models.unet import UNet
from tests.geospatial.conftest import write_tif

STATS = NormStats(mean=(-10.0, -17.0), std=(4.0, 4.5), clip=(-50.0, 5.0), n_pixels=1, source="t")


@pytest.mark.parametrize("h,w", [(10, 10), (100, 37), (513, 1025), (512, 512)])
def test_tile_grid_covers_every_pixel(h, w):
    cov = np.zeros((h, w), int)
    for s in tile_grid(h, w, tile=128, overlap=16):
        cov[s.row : s.row + s.height, s.col : s.col + s.width] += 1
        assert s.row >= 0 and s.col >= 0
    assert cov.min() >= 1


def test_tile_grid_rejects_bad_overlap():
    with pytest.raises(ValueError):
        tile_grid(100, 100, tile=32, overlap=16)


def test_hann_window_tapers_only_interior_sides():
    w = hann_window(64, 8)
    assert w.shape == (64, 64) and w[32, 32] == pytest.approx(1.0)
    assert w[0, 32] == 0.0 and w[32, 0] == 0.0  # dead band at interior edges
    edge = hann_window(64, 8, top=False, left=False)
    assert edge[0, 32] == 1.0 and edge[32, 0] == 1.0 and edge[63, 32] == 0.0


def _reader(img):
    def read(s, size):
        out = np.zeros((img.shape[0], size, size), np.float32)
        out[:, : s.height, : s.width] = img[:, s.row : s.row + s.height, s.col : s.col + s.width]
        return out

    return read


def test_blending_reproduces_identity_model_exactly(tmp_path):
    img = np.random.default_rng(0).random((2, 300, 410)).astype(np.float32)
    out = tiled_predict(
        _reader(img), lambda x: x[:, 0], 300, 410, tile=128, overlap=16, batch=3, workdir=tmp_path
    )
    np.testing.assert_allclose(np.asarray(out), img[0], atol=1e-5)


def test_tiled_matches_full_image_for_conv_model(tmp_path):
    torch.manual_seed(0)
    conv = nn.Conv2d(2, 1, 5, padding=2)
    img = np.random.default_rng(1).normal(size=(2, 256, 256)).astype(np.float32)

    def predict(x):
        with torch.no_grad():
            return torch.sigmoid(conv(torch.from_numpy(x)))[:, 0].numpy()

    full = predict(img[None])[0]
    tiled = np.asarray(
        tiled_predict(
            _reader(img), predict, 256, 256, tile=96, overlap=16, batch=4, workdir=tmp_path
        )
    )
    # Interior tile edges get zero weight, so the result matches full-image inference everywhere.
    assert np.abs(tiled - full).max() < 1e-5


class DarkIsWater(nn.Module):
    """Deterministic 'model': water where normalised VH is low (dark)."""

    def forward(self, x):
        return -x[:, 1:2] * 20.0 - 10.0


@pytest.fixture
def scene(tmp_path):
    h, w = 300, 520
    vv = np.full((h, w), -8.0, np.float32)
    vh = np.full((h, w), -15.0, np.float32)
    vv[100:200, 50:250] = -22.0  # 100 × 200 px open water
    vh[100:200, 50:250] = -28.0
    img = np.stack([vv, vh])
    img[:, :20, :] = np.nan  # nodata strip
    tf = from_origin(500000, 4600000, 10, 10)  # UTM 33N, 10 m pixels
    path = write_tif(tmp_path / "s1.tif", img, crs="EPSG:32633", transform=tf)
    with rasterio.open(path, "r+") as d:
        d.descriptions = ("VV", "VH")
    return path


def test_predictor_end_to_end(scene, tmp_path):
    fp = FloodPredictor(DarkIsWater(), STATS, torch.device("cpu"), 0.5, "test-model")
    progress = []
    out = tmp_path / "out"
    summary = fp.run(scene, out, tile=128, overlap=16, batch=2, progress=progress.append)
    assert progress[-1] == pytest.approx(1.0)
    assert summary["pixels_flooded"] == 100 * 200
    assert summary["flooded_area_km2"] == pytest.approx(100 * 200 * 100 / 1e6)
    assert summary["pixels_valid"] == (300 - 20) * 520
    with rasterio.open(out / "flood_mask.tif") as m, rasterio.open(scene) as s:
        assert m.crs == s.crs and m.transform == s.transform
        mask = m.read(1)
        assert m.tags(ns="IMAGE_STRUCTURE").get("LAYOUT") == "COG"
    assert (mask[:20] == NODATA_U8).all()
    assert mask[150, 100] == 1 and mask[250, 400] == 0
    gj = json.loads((out / "flood_polygons.geojson").read_text())
    assert len(gj["features"]) == 1
    assert gj["features"][0]["properties"]["area_m2"] == pytest.approx(2_000_000, rel=0.01)
    lon, lat = gj["features"][0]["geometry"]["coordinates"][0][0]
    assert 14 < lon < 16 and 41 < lat < 42  # reprojected to WGS84
    assert not (out / ".tiles").exists()


def test_predictor_from_checkpoint(scene, tmp_path):
    m = UNet(2, 1, base_channels=4, depth=2)
    ckpt = tmp_path / "m.pt"
    torch.save(
        {
            "model": m.state_dict(),
            "config": {
                "arch": "unet",
                "base_channels": 4,
                "depth": 2,
                "dropout": 0.0,
                "threshold": 0.5,
            },
            "meta": {"norm_stats": STATS.__dict__, "run_name": "tiny"},
        },
        ckpt,
    )
    fp = FloodPredictor.from_checkpoint(ckpt, device="cpu")
    s = fp.run(scene, tmp_path / "o", tile=64, overlap=8)
    assert s["model"] == "tiny" and 0 <= s["pixels_flooded"] <= s["pixels_valid"]


def test_validate_input_rejections(tmp_path):
    with pytest.raises(InputValidationError, match="2 bands"):
        validate_input(write_tif(tmp_path / "a.tif", np.zeros((3, 8, 8), np.float32)))
    with pytest.raises(InputValidationError, match="float"):
        validate_input(write_tif(tmp_path / "b.tif", np.zeros((2, 8, 8), np.uint8)))
    p = tmp_path / "c.tif"
    p.write_bytes(b"junk")
    with pytest.raises(InputValidationError, match="readable"):
        validate_input(p)
    with pytest.raises(InputValidationError, match="too large"):
        validate_input(
            write_tif(tmp_path / "d.tif", np.zeros((2, 8, 8), np.float32)), max_pixels=10
        )
