from datetime import date
from types import SimpleNamespace

import numpy as np
import pytest

from geospatial.stac import search_sentinel1, validate_bbox
from geospatial.viz import nodata_hatch, overlay_mask, sar_false_color


def test_false_color_shapes_and_nan():
    vv = np.array([[-30.0, 0.0], [np.nan, -12.5]], np.float32)
    vh = np.array([[-35.0, -5.0], [np.nan, -17.5]], np.float32)
    rgb = sar_false_color(vv, vh)
    assert rgb.shape == (2, 2, 3) and rgb.dtype == np.uint8
    assert (rgb[1, 0] == 0).all()  # NaN -> black
    assert rgb[0, 1, 0] == 255 and rgb[0, 0, 0] == 0  # VV stretch endpoints


def test_overlay_only_colors_water():
    rgb = np.zeros((2, 2, 3), np.uint8)
    out = overlay_mask(rgb, np.array([[1, 0], [-1, 1]]))
    assert out[0, 0, 2] > 0 and (out[0, 1] == 0).all() and (out[1, 0] == 0).all()
    hatched = nodata_hatch(rgb, np.ones((8, 8), bool)[:2, :2])
    assert (hatched[0, 0] == [255, 0, 255]).all()


def test_bbox_validation():
    with pytest.raises(ValueError):
        validate_bbox((10, 0, 5, 1))
    with pytest.raises(ValueError):
        validate_bbox((0, -95, 1, 0))


class FakeClient:
    def __init__(self):
        self.kwargs = None

    def search(self, **kwargs):
        self.kwargs = kwargs
        item = SimpleNamespace(
            id="S1A_X",
            bbox=[0, 0, 1, 1],
            assets={"vv": 1, "vh": 2},
            properties={
                "datetime": "2019-01-01T00:00:00Z",
                "platform": "sentinel-1a",
                "sar:instrument_mode": "IW",
                "sar:polarizations": ["VV", "VH"],
            },
        )
        return SimpleNamespace(items=lambda: iter([item]))


def test_search_sentinel1_builds_query_and_summarizes():
    c = FakeClient()
    res = search_sentinel1((0, 0, 1, 1), date(2019, 1, 1), date(2019, 1, 31), client=c, max_items=5)
    assert c.kwargs["datetime"] == "2019-01-01/2019-01-31"
    assert c.kwargs["collections"] == ["sentinel-1-grd"] and c.kwargs["max_items"] == 5
    assert res[0].polarizations == ("VV", "VH") and res[0].assets == ("vh", "vv")


def test_search_rejects_reversed_dates():
    with pytest.raises(ValueError):
        search_sentinel1((0, 0, 1, 1), date(2019, 2, 1), date(2019, 1, 1), client=FakeClient())
