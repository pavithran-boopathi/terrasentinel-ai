import math

import numpy as np
import pytest
import torch

from ml.evaluation.metrics import (
    IGNORE_INDEX,
    MetricAccumulator,
    bootstrap_iou_ci,
    confusion,
    scores,
)
from ml.models.unet import UNet, count_parameters
from ml.training.augment import train_transform
from ml.training.losses import BCEDiceLoss


def test_confusion_and_scores_by_hand():
    pred = torch.tensor([[1, 1, 0, 0, 1]])
    tgt = torch.tensor([[1, 0, 1, 0, IGNORE_INDEX]])
    tp, fp, fn, tn = confusion(pred, tgt)
    assert (tp, fp, fn, tn) == (1, 1, 1, 1)  # ignored pixel excluded even though pred=1
    s = scores(tp, fp, fn, tn)
    assert s["iou"] == pytest.approx(1 / 3)
    assert s["dice"] == pytest.approx(0.5) == s["f1"]
    assert s["precision"] == s["recall"] == 0.5
    assert s["accuracy"] == 0.5


def test_scores_undefined_is_nan():
    s = scores(0, 0, 0, 10)
    assert math.isnan(s["iou"]) and math.isnan(s["precision"])
    assert s["accuracy"] == 1.0


def test_accumulator_pools_pixels_not_chips():
    acc = MetricAccumulator()
    pred = torch.tensor([[[1, 1], [1, 1]], [[0, 0], [0, 0]]])
    tgt = torch.tensor([[[1, 1], [1, 1]], [[1, 0], [0, 0]]])
    acc.update(pred, tgt, chips=["a", "b"])
    m = acc.compute()
    assert m["iou"] == pytest.approx(4 / 5)  # pooled: tp=4, fn=1
    assert m["mean_chip_iou"] == pytest.approx((1.0 + 0.0) / 2)
    assert acc.per_chip["b"]["n_water"] == 1


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    counts = [(int(t), int(f), int(n)) for t, f, n in rng.integers(50, 200, (40, 3))]
    tp, fp, fn = np.sum(counts, axis=0)
    point = tp / (tp + fp + fn)
    lo, hi = bootstrap_iou_ci(counts, n_boot=500)
    assert lo < point < hi and hi - lo < 0.2


def test_loss_ignores_ignore_pixels():
    loss = BCEDiceLoss()
    logits = torch.zeros(1, 1, 2, 2, requires_grad=True)
    tgt = torch.tensor([[[1, 0], [IGNORE_INDEX, IGNORE_INDEX]]])
    base = loss(logits, tgt)
    huge = logits.detach().clone()
    huge[0, 0, 1, :] = 1e4  # wildly wrong on ignored pixels only
    assert loss(huge, tgt).item() == pytest.approx(base.item())
    base.backward()
    assert logits.grad[0, 0, 1].abs().sum() == 0


def test_loss_all_ignored_is_zero_with_grad():
    logits = torch.randn(1, 1, 2, 2, requires_grad=True)
    out = BCEDiceLoss()(logits, torch.full((1, 2, 2), IGNORE_INDEX))
    out.backward()
    assert out.item() == 0.0


def test_loss_decreases_for_better_predictions():
    tgt = torch.tensor([[[1, 0], [0, 1]]])
    good = torch.tensor([[[[5.0, -5.0], [-5.0, 5.0]]]])
    bad = -good
    lf = BCEDiceLoss(pos_weight=2.0)
    assert lf(good, tgt) < lf(bad, tgt)


def test_unet_shapes_and_validation():
    m = UNet(base_channels=8, depth=3, dropout=0.1)
    out = m(torch.randn(2, 2, 64, 64))
    assert out.shape == (2, 1, 64, 64)
    assert count_parameters(m) > 0
    with pytest.raises(ValueError, match="divisible"):
        m(torch.randn(1, 2, 60, 60))


def test_unet_overfits_one_batch():
    torch.manual_seed(0)
    x = torch.randn(2, 2, 32, 32)
    y = (x[:, 0] > 0).long()
    m = UNet(base_channels=8, depth=2)
    opt = torch.optim.Adam(m.parameters(), 1e-2)
    lf = BCEDiceLoss()
    first = lf(m(x), y).item()
    for _ in range(60):
        opt.zero_grad()
        loss = lf(m(x), y)
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.3


def test_augment_keeps_image_mask_aligned():
    img = np.zeros((64, 64, 2), np.float32)
    img[:10, :20, 0] = 1.0
    mask = (img[..., 0] > 0).astype(np.uint8)
    mask[40:, 40:] = IGNORE_INDEX
    t = train_transform(crop=None, noise_std=0.0)
    for seed in range(10):
        t.set_random_seed(seed)
        out = t(image=img, mask=mask)
        water = out["mask"] == 1
        assert np.array_equal(water, out["image"][..., 0] > 0.5)
        assert set(np.unique(out["mask"])) <= {0, 1, IGNORE_INDEX}
