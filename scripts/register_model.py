#!/usr/bin/env python3
"""Register a trained run in the model registry (copies weights into object storage).

    python scripts/register_model.py --run data/checkpoints/<run> --name flood-unet --version 1 [--default]

Metrics recorded are copied verbatim from the run's results.json — nothing is typed by hand.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sqlalchemy import select, update  # noqa: E402

import backend.models  # noqa: E402,F401
from backend.db.session import get_sessionmaker  # noqa: E402
from backend.models.registry import ModelVersion  # noqa: E402
from backend.services.storage import get_storage, sha256_file  # noqa: E402


def metrics_from_results(results: dict) -> dict:
    out = {}
    for split, r in results["splits"].items():
        p = r["pooled"]
        out[split] = {k: p[k] for k in ("iou", "f1", "precision", "recall") if k in p}
        out[split]["iou_ci95"] = p.get("iou_ci95")
        out[split]["chips"] = len(r.get("per_chip", {}))
    if "latency" in results:
        out["latency"] = {
            k: {"ms": v["latency_ms_mean"], "device": v["device"]}
            for k, v in results["latency"].items()
        }
    if "threshold" in results:
        out["val_tuned_threshold"] = results["threshold"]["val_tuned"]
    return out


def register(run: Path, name: str, version: str, default: bool, tuned_threshold: bool) -> str:
    results = json.loads((run / "results.json").read_text())
    cfg = results["config"]
    storage = get_storage()
    key = f"models/{name}/{version}/model.pt"
    uri = storage.put_file(key, run / "best.pt")
    path = storage.local_path(uri)
    thr = (
        results["threshold"]["val_tuned"]
        if tuned_threshold and "threshold" in results
        else cfg.get("threshold", 0.5)
    )
    with get_sessionmaker()() as db:
        if db.scalar(
            select(ModelVersion).where(ModelVersion.name == name, ModelVersion.version == version)
        ):
            raise SystemExit(f"{name}:{version} already registered")
        if default:
            db.execute(update(ModelVersion).values(is_default=False))
        mv = ModelVersion(
            name=name,
            version=version,
            arch=cfg.get("arch", "unet"),
            artifact_uri=uri,
            sha256=sha256_file(path),
            size_bytes=path.stat().st_size,
            threshold=float(thr),
            metrics=metrics_from_results(results),
            is_default=default,
            training={
                "run": results["meta"]["run_name"],
                "split_scheme": cfg["split_scheme"],
                "split_sha256": results["meta"]["split_sha256"],
                "seed": cfg["seed"],
                "git_commit": results["meta"]["git"]["commit"],
                "params": results["meta"]["params"],
                "best_epoch": results["best_epoch"],
                "device": results["meta"]["device"],
            },
        )
        db.add(mv)
        db.commit()
        return str(mv.id)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--default", action="store_true")
    ap.add_argument(
        "--tuned-threshold",
        action="store_true",
        help="use the threshold selected on the validation split",
    )
    a = ap.parse_args()
    print(register(a.run, a.name, a.version, a.default, a.tuned_threshold))


if __name__ == "__main__":
    main()
