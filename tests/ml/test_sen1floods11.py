import csv

import numpy as np
import pytest
import torch

from ml.data.sen1floods11 import (
    HAND,
    IGNORE_INDEX,
    SPLIT_DIR,
    Chip,
    Sen1Floods11Dataset,
    check_no_leakage,
    compute_norm_stats,
    discover_chips,
    encode_target,
    event_holdout_splits,
    label_summary,
    leave_one_event_out,
    load_splits_json,
    normalize,
    official_splits,
    read_chip,
    splits_to_json,
)
from tests.geospatial.conftest import write_tif

EVENTS = {"Bolivia": 2, "Ghana": 3, "Sri-Lanka": 3, "USA": 4}


@pytest.fixture
def root(tmp_path):
    """Miniature Sen1Floods11 tree with 16×16 chips."""
    rng = np.random.default_rng(0)
    names = []
    for ev, n in EVENTS.items():
        for i in range(n):
            name = f"{ev}_{1000 + i}"
            names.append(name)
            img = rng.normal([[[-12.0]], [[-20.0]]], 2.0, (2, 16, 16)).astype(np.float32)
            lab = (img[0] < -12).astype(np.int16)
            lab[0, :] = -1
            img[:, 1, 0] = np.nan  # S1 nodata inside a labelled row
            write_tif(tmp_path / HAND / "S1Hand" / f"{name}_S1Hand.tif", img)
            write_tif(tmp_path / HAND / "LabelHand" / f"{name}_LabelHand.tif", lab)
    split_dir = tmp_path / SPLIT_DIR
    split_dir.mkdir(parents=True)
    non_bo = [n for n in names if not n.startswith("Bolivia")]
    parts = {"train": non_bo[:6], "valid": non_bo[6:8], "test": non_bo[8:], "bolivia": names[:2]}
    for k, v in parts.items():
        with (split_dir / f"flood_{k}_data.csv").open("w", newline="") as f:
            csv.writer(f).writerows([[f"{n}_S1Hand.tif", f"{n}_LabelHand.tif"] for n in v])
    return tmp_path


def test_chip_parsing_handles_hyphenated_events():
    c = Chip.from_filename("Sri-Lanka_101_S1Hand.tif")
    assert (c.event, c.chip_id, c.name) == ("Sri-Lanka", "101", "Sri-Lanka_101")
    assert Chip.from_name(c.name) == c
    with pytest.raises(ValueError):
        Chip.from_filename("bad.tif")
    with pytest.raises(ValueError):
        Chip.from_name("no-underscore")


def test_discover_requires_both_files(root):
    (root / HAND / "LabelHand" / "USA_1003_LabelHand.tif").unlink()
    chips = discover_chips(root)
    assert len(chips) == sum(EVENTS.values()) - 1
    assert Chip("USA", "1003") not in chips


def test_official_splits(root):
    s = official_splits(root)
    assert {k: len(v) for k, v in s.items()} == {"train": 6, "val": 2, "test": 2, "bolivia": 2}
    assert check_no_leakage(s, by_event=False) == []
    assert check_no_leakage(s, by_event=True)  # official splits share events — the leakage we avoid


def test_event_holdout_has_no_shared_events(root):
    s = event_holdout_splits(
        discover_chips(root), test_events=["Bolivia", "Ghana"], val_events=["USA"]
    )
    assert {c.event for c in s["test"]} == {"Bolivia", "Ghana"}
    assert {c.event for c in s["train"]} == {"Sri-Lanka"}
    assert check_no_leakage(s, by_event=True) == []
    with pytest.raises(ValueError, match="unknown"):
        event_holdout_splits(discover_chips(root), ["Atlantis"], [])
    with pytest.raises(ValueError, match="both"):
        event_holdout_splits(discover_chips(root), ["USA"], ["USA"])


def test_leave_one_event_out(root):
    folds = leave_one_event_out(discover_chips(root))
    assert [f["held_out"] for f in folds] == sorted(EVENTS)
    for f in folds:
        assert all(c.event == f["held_out"] for c in f["test"])
        assert all(c.event != f["held_out"] for c in f["train"])


def test_split_json_roundtrip_and_tamper_detection(root, tmp_path):
    s = event_holdout_splits(discover_chips(root), ["Bolivia"], ["USA"])
    text, digest = splits_to_json(s, {"name": "t"})
    p = tmp_path / "s.json"
    p.write_text(text)
    assert load_splits_json(p) == s
    assert splits_to_json(s, {"name": "t"})[1] == digest  # deterministic
    p.write_text(text.replace("Ghana_1000", "Ghana_9999"))
    with pytest.raises(ValueError, match="hash"):
        load_splits_json(p)


def test_encode_target_masks_label_and_s1_nodata():
    raw = np.array([[-1, 0], [1, 1]], np.int16)
    img = np.zeros((2, 2, 2), np.float32)
    img[1, 1, 1] = np.nan
    t = encode_target(raw, img)
    assert t.tolist() == [[IGNORE_INDEX, 0], [1, IGNORE_INDEX]]


def test_norm_stats_match_numpy(root):
    chips = official_splits(root)["train"]
    stats = compute_norm_stats(chips, root, source="test")
    imgs = np.stack([read_chip(c, root)[0] for c in chips])
    valid = np.isfinite(imgs).all(axis=1)
    vv = imgs[:, 0][valid]
    assert stats.mean[0] == pytest.approx(vv.mean(), abs=1e-4)
    assert stats.std[0] == pytest.approx(vv.std(), abs=1e-4)
    assert stats.n_pixels == valid.sum()
    norm = normalize(imgs[0], stats)
    assert np.isfinite(norm).all() and norm.dtype == np.float32


def test_label_summary(root):
    s = label_summary(official_splits(root)["train"], root)
    assert s["ignore"] == 6 * (16 + 1)  # first row + one NaN pixel per chip
    assert 0 < s["water_fraction_of_labelled"] < 1


def test_dataset_returns_tensors_and_applies_transform(root):
    chips = discover_chips(root)
    stats = compute_norm_stats(chips, root, source="test")
    ds = Sen1Floods11Dataset(chips, root, stats)
    item = ds[0]
    assert item["image"].shape == (2, 16, 16) and item["image"].dtype == torch.float32
    assert item["mask"].dtype == torch.int64
    assert set(item["mask"].unique().tolist()) <= {0, 1, IGNORE_INDEX}

    def hflip(image, mask):
        return {"image": image[:, ::-1], "mask": mask[:, ::-1]}

    flipped = Sen1Floods11Dataset(chips, root, stats, transform=hflip)[0]
    assert torch.equal(flipped["mask"], item["mask"].flip(-1))
    assert torch.allclose(flipped["image"], item["image"].flip(-1))
