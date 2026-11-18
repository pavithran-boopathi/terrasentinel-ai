import numpy as np
import pytest
import torch
from torch import nn

from ml.evaluation.calibration import brier, ece, fit_temperature, nll, reliability, subsample
from ml.evaluation.metrics import IGNORE_INDEX
from ml.evaluation.uncertainty import (
    error_detection_auroc,
    mc_dropout_probs,
    risk_coverage,
    summarize_samples,
    tta_probs,
)


def test_perfectly_calibrated_has_near_zero_ece():
    rng = np.random.default_rng(0)
    p = rng.random(200_000)
    y = (rng.random(200_000) < p).astype(np.uint8)
    assert ece(p, y) < 0.01
    r = reliability(p, y, 10)
    assert np.allclose(r.confidence, r.accuracy, atol=0.02)


def test_overconfident_model_has_high_ece_and_ignore_is_excluded():
    y = np.array([1, 0] * 500, np.uint8)
    p = np.where(y == 1, 0.99, 0.01).astype(float)
    p[::4] = 1 - p[::4]  # 25% confidently wrong
    assert ece(p, y) > 0.2
    y2 = y.copy()
    y2[::4] = IGNORE_INDEX
    assert ece(p, y2) < 0.02 and brier(p, y2) < 0.001


def test_temperature_scaling_recovers_true_temperature():
    rng = np.random.default_rng(1)
    z_true = rng.normal(0, 2, 100_000)
    y = (rng.random(z_true.size) < 1 / (1 + np.exp(-z_true))).astype(np.uint8)
    t = fit_temperature(z_true * 3.0, y)  # model 3× overconfident
    assert t == pytest.approx(3.0, rel=0.05)
    before = nll(1 / (1 + np.exp(-z_true * 3)), y)
    after = nll(1 / (1 + np.exp(-z_true * 3 / t)), y)
    assert after < before


def test_subsample_balances_chips():
    a = [np.arange(100.0).reshape(10, 10), np.arange(100.0).reshape(10, 10)]
    y = [np.zeros((10, 10), np.uint8), np.full((10, 10), IGNORE_INDEX, np.uint8)]
    y[1][0, :3] = 1
    xs, ys = subsample(a, y, per_chip=5)
    assert len(xs) == 5 + 3 and (ys != IGNORE_INDEX).all()


class Tiny(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.conv = nn.Conv2d(2, 4, 3, padding=1)
        self.drop = nn.Dropout2d(p)
        self.head = nn.Conv2d(4, 1, 1)

    def forward(self, x):
        return self.head(self.drop(torch.relu(self.conv(x))))


def test_mc_dropout_produces_varied_samples_and_requires_dropout():
    torch.manual_seed(0)
    m = Tiny()
    s = mc_dropout_probs(m, torch.randn(1, 2, 16, 16), passes=6)
    assert s.shape == (6, 1, 16, 16) and s.std(0).mean() > 0
    assert not m.drop.training  # restored to eval
    with pytest.raises(ValueError, match="no dropout"):
        mc_dropout_probs(nn.Conv2d(2, 1, 1), torch.randn(1, 2, 4, 4))


def test_tta_is_exact_for_equivariant_model():
    m = nn.Conv2d(2, 1, 1)  # 1×1 conv commutes with rotations/flips
    x = torch.randn(1, 2, 8, 8)
    s = tta_probs(m, x)
    assert s.shape == (8, 1, 8, 8)
    assert torch.allclose(s, s[0].expand_as(s), atol=1e-6)
    out = summarize_samples(s)
    assert out["std"].max() < 1e-6 and out["entropy"].shape == (1, 8, 8)


def test_error_detection_auroc_and_risk_coverage():
    y = np.array([1, 1, 0, 0, 1, 0, IGNORE_INDEX], np.uint8)
    p = np.array([0.9, 0.4, 0.1, 0.6, 0.8, 0.2, 0.9])
    err_unc = np.array([0.1, 0.9, 0.1, 0.8, 0.2, 0.1, 0.5])  # high exactly on the 2 errors
    assert error_detection_auroc(err_unc, p, y) == pytest.approx(1.0)
    assert error_detection_auroc(-err_unc, p, y) == pytest.approx(0.0)
    rc = risk_coverage(err_unc, p, y, coverages=(1.0, 0.67))
    assert rc[0.67] == pytest.approx(1.0) and rc[1.0] < 1.0
