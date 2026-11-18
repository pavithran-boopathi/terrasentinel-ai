import json

import numpy as np
import pytest
from PIL import Image

from ml.data.xbd import IGNORE, XBDDataset, event_split, index_tiles, rasterize_damage, xview2_score
from ml.training.train_xbd import run

pytest.importorskip("segmentation_models_pytorch")


def label(polys):
    return {
        "features": {
            "xy": [
                {"wkt": w, "properties": {"feature_type": "building", "subtype": s}}
                for w, s in polys
            ]
        }
    }


def test_rasterize_damage_classes_and_unclassified():
    lj = label(
        [
            ("POLYGON ((2 2, 10 2, 10 10, 2 10, 2 2))", "destroyed"),
            ("POLYGON ((20 20, 30 20, 30 30, 20 30, 20 20))", "no-damage"),
            ("POLYGON ((40 2, 50 2, 50 10, 40 10, 40 2))", "un-classified"),
        ]
    )
    m = rasterize_damage(lj, (64, 64))
    assert m[5, 5] == 4 and m[25, 25] == 1 and m[5, 45] == IGNORE and m[60, 60] == 0


def test_xview2_score_perfect_and_bounds():
    conf = np.diag([1000, 50, 20, 20, 10])
    s = xview2_score(conf)
    assert s["xview2_score"] == pytest.approx(1.0) and s["f1_localization"] == pytest.approx(1.0)
    conf2 = conf.copy()
    conf2[4, 4], conf2[4, 1] = 0, 10  # all destroyed predicted as no-damage
    assert xview2_score(conf2)["f1_destroyed"] == 0.0
    assert xview2_score(conf2)["xview2_score"] < 0.4  # harmonic mean punishes a missed class


def make_fake_xbd(root, disasters=("flood-a", "fire-b", "quake-c"), n=2, size=64):
    rng = np.random.default_rng(0)
    for split in ("tier1",):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        for d in disasters:
            for i in range(n):
                stem = f"{d}_{i:08d}"
                for phase in ("pre", "post"):
                    arr = rng.integers(0, 255, (size, size, 3), np.uint8)
                    Image.fromarray(arr).save(
                        root / split / "images" / f"{stem}_{phase}_disaster.png"
                    )
                lj = label(
                    [
                        ("POLYGON ((8 8, 24 8, 24 24, 8 24, 8 8))", "destroyed"),
                        ("POLYGON ((36 36, 56 36, 56 56, 36 56, 36 36))", "no-damage"),
                    ]
                )
                (root / split / "labels" / f"{stem}_post_disaster.json").write_text(json.dumps(lj))


def test_index_split_dataset(tmp_path):
    make_fake_xbd(tmp_path)
    tiles = index_tiles(tmp_path)
    assert len(tiles) == 6 and {t.disaster for t in tiles} == {"flood-a", "fire-b", "quake-c"}
    s = event_split(tiles, {"quake-c"}, {"fire-b"})
    assert {t.disaster for t in s["test"]} == {"quake-c"} and len(s["train"]) == 2
    item = XBDDataset(s["train"], crop=32)[0]
    assert item["image"].shape == (6, 32, 32) and item["mask"].max() <= 4


def test_training_pipeline_runs_end_to_end_on_synthetic_xbd(tmp_path):
    make_fake_xbd(tmp_path / "xbd")
    res = run(
        tmp_path / "xbd",
        {"quake-c"},
        {"fire-b"},
        epochs=1,
        crop=32,
        batch=2,
        lr=1e-3,
        seed=0,
        pretrained=False,
        out_root=tmp_path / "ckpt",
        device="cpu",
    )
    assert set(res["test"]) >= {"xview2_score", "f1_localization", "f1_destroyed"}
    assert 0.0 <= res["test"]["xview2_score"] <= 1.0
    assert len(list((tmp_path / "ckpt").glob("*/best.pt"))) == 1
