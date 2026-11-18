"""End-to-end smoke test of training on a synthetic Sen1Floods11 tree (CPU, no MLflow)."""

import json

import pytest

from ml.data.sen1floods11 import (
    compute_norm_stats,
    discover_chips,
    event_holdout_splits,
    splits_to_json,
)
from ml.training.config import TrainConfig
from ml.training.train import run
from tests.ml.test_sen1floods11 import root  # noqa: F401 - fixture


@pytest.fixture
def prepared(root, tmp_path):  # noqa: F811
    proc = tmp_path / "proc"
    (proc / "splits").mkdir(parents=True)
    sp = event_holdout_splits(discover_chips(root), ["Bolivia"], ["Ghana"])
    text, _ = splits_to_json(sp, {"name": "geo"})
    (proc / "splits" / "geo.json").write_text(text)
    (proc / "norm_stats_geo.json").write_text(
        compute_norm_stats(sp["train"], root, "geo/train").to_json()
    )
    return root, proc, tmp_path / "ckpt"


def test_training_runs_and_writes_results(prepared):
    raw, proc, out = prepared
    cfg = TrainConfig(
        name="smoke",
        split_scheme="geo",
        raw_root=str(raw),
        processed_root=str(proc),
        eval_splits=["test"],
        base_channels=4,
        depth=2,
        epochs=2,
        batch_size=2,
        crop=None,
        device="cpu",
        mlflow=False,
        output_root=str(out),
        latency_size=32,
        warmup_epochs=1,
    )
    res = run(cfg)
    run_dir = next(out.iterdir())
    assert (run_dir / "best.pt").exists()
    saved = json.loads((run_dir / "results.json").read_text())
    assert set(saved["splits"]) == {"val", "test"}
    assert saved["meta"]["split_sha256"]
    assert {c.split("_")[0] for c in saved["splits"]["test"]["per_chip"]} == {"Bolivia"}
    assert len(json.loads((run_dir / "history.json").read_text())) == 2
    assert 0.0 <= res["splits"]["test"]["pooled"]["accuracy"] <= 1.0
    assert res["latency"]["bs1"]["latency_ms_mean"] > 0


def test_config_rejects_unknown_keys(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("name: x\nlearning_rate: 0.1\n")
    with pytest.raises(ValueError, match="learning_rate"):
        TrainConfig.from_yaml(p)
    p.write_text("name: x\nepochs: 3\n")
    assert TrainConfig.from_yaml(p, epochs=5).epochs == 5
