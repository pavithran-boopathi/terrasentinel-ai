import numpy as np
import pytest

from geospatial.change import (
    NODATA,
    darkening_mask,
    log_ratio,
    new_water,
    receded_water,
    split_permanent,
)


def scene(vh_water_rows):
    img = np.stack([np.full((6, 6), -8.0), np.full((6, 6), -15.0)]).astype(np.float32)
    img[1, vh_water_rows] = -25.0
    img[0, vh_water_rows] = -20.0
    return img


def test_log_ratio_and_darkening():
    pre, post = scene(slice(0, 1)), scene(slice(0, 3))  # river (row 0) widens to rows 0–2
    d = log_ratio(pre, post)
    assert d[1, 1, 0] == pytest.approx(-10.0) and d[1, 0, 0] == 0.0
    m = darkening_mask(pre, post)
    assert m[0].sum() == 0  # already water before → not a change
    assert m[1:3].all() and not m[3:].any()
    post[:, 5, 5] = np.nan
    assert darkening_mask(pre, post)[5, 5] == NODATA
    with pytest.raises(ValueError):
        log_ratio(pre, post[:, :3])


def test_new_and_receded_water_with_nodata():
    pre = np.array([[1, 0, 0, NODATA]], np.uint8)
    post = np.array([[1, 1, 0, 1]], np.uint8)
    assert new_water(pre, post).tolist() == [[0, 1, 0, NODATA]]
    assert receded_water(pre, post).tolist() == [[0, 0, 0, NODATA]]


def test_split_permanent():
    water = np.array([[1, 1, 0, NODATA]], np.uint8)
    occ = np.array([[90, 10, 95, 0]], np.float32)
    flood, perm = split_permanent(water, occ)
    assert flood.tolist() == [[0, 1, 0, NODATA]] and perm.tolist() == [[1, 0, 0, NODATA]]
