import numpy as np
import pytest

from ml.baselines.classical import (
    GlobalThreshold,
    PixelClassifier,
    otsu_predict,
    otsu_threshold,
    pixel_features,
)
from ml.data.sen1floods11 import IGNORE_INDEX


def bimodal_chip(seed=0, n=32):
    rng = np.random.default_rng(seed)
    water = np.zeros((n, n), bool)
    water[: n // 2] = True
    vh = np.where(water, rng.normal(-25, 1.5, (n, n)), rng.normal(-14, 1.5, (n, n)))
    vv = vh + 7
    img = np.stack([vv, vh]).astype(np.float32)
    return img, water.astype(np.uint8)


def test_otsu_separates_bimodal_histogram():
    vals = np.concatenate([np.full(100, -25.0), np.full(100, -14.0)])
    assert -25 < otsu_threshold(vals) < -14
    with pytest.raises(ValueError):
        otsu_threshold(np.array([np.nan]))
    assert otsu_threshold(np.array([3.0, 3.0])) == 3.0


def test_otsu_predict_recovers_water_and_masks_nan():
    img, water = bimodal_chip()
    img[:, 0, 0] = np.nan
    pred = otsu_predict(img)
    assert pred[0, 0] == 0
    assert (pred[1:] == water[1:]).mean() > 0.95


def test_global_threshold_learns_between_modes():
    imgs, tgts = zip(*[bimodal_chip(s) for s in range(3)], strict=True)
    tgts = [t.copy() for t in tgts]
    tgts[0][-2:] = IGNORE_INDEX
    gt = GlobalThreshold().fit(list(imgs), tgts)
    assert -24 < gt.threshold < -15
    assert (gt.predict(imgs[1]) == tgts[1]).mean() > 0.95
    with pytest.raises(RuntimeError):
        GlobalThreshold().predict(imgs[0])


def test_pixel_features_shape_and_nan_fill():
    img, _ = bimodal_chip(n=16)
    img[0, 3, 3] = np.nan
    f = pixel_features(img)
    assert f.shape == (256, 3 + 2 * 2 * 2) and np.isfinite(f).all()


def test_pixel_classifier_fits_and_predicts():
    imgs, tgts = zip(*[bimodal_chip(s) for s in range(4)], strict=True)
    pc = PixelClassifier(samples_per_chip=500).fit(list(imgs), list(tgts))
    test_img, test_t = bimodal_chip(9)
    assert (pc.predict(test_img) == test_t).mean() > 0.95
