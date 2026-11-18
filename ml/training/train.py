"""Train a flood-segmentation U-Net on Sen1Floods11.

    python -m ml.training.train --config ml/configs/unet_official.yaml [--epochs 2 --no-mlflow]

Writes ``<output_root>/<run>/`` with ``best.pt``, ``history.json`` and ``results.json`` and logs
params/metrics/artifacts to MLflow. Every reported number comes from ``results.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ml.data.sen1floods11 import IGNORE_INDEX, NormStats, Sen1Floods11Dataset, load_splits_json
from ml.evaluation.metrics import bootstrap_iou_ci, group_scores
from ml.models.factory import build_model
from ml.models.unet import count_parameters
from ml.training.augment import train_transform
from ml.training.config import TrainConfig
from ml.training.engine import (
    benchmark_latency,
    cosine_with_warmup,
    evaluate,
    seed_everything,
    select_device,
    sweep_thresholds,
    train_one_epoch,
)
from ml.training.losses import BCEDiceLoss

REPO = Path(__file__).resolve().parents[2]


def git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.run(  # noqa: S603
                ["git", *args],  # noqa: S607
                cwd=REPO,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else REPO / path


def build_datasets(cfg: TrainConfig):
    proc = resolve(cfg.processed_root)
    split_file = proc / "splits" / f"{cfg.split_scheme}.json"
    splits = load_splits_json(split_file)
    split_sha = json.loads(split_file.read_text())["meta"]["sha256"]
    stats = NormStats.from_json((proc / f"norm_stats_{cfg.split_scheme}.json").read_text())
    root = resolve(cfg.raw_root)
    tf = train_transform(cfg.crop, cfg.noise_std)
    ds = {"train": Sen1Floods11Dataset(splits["train"], root, stats, transform=tf)}
    for name in {"val", *cfg.eval_splits}:
        ds[name] = Sen1Floods11Dataset(splits[name], root, stats)
    if cfg.preload:
        for d in ds.values():
            d.preload()
    return ds, stats, split_sha


def resolve_pos_weight(cfg: TrainConfig, train_ds: Sen1Floods11Dataset) -> float | None:
    if cfg.pos_weight is None or isinstance(cfg.pos_weight, int | float):
        return cfg.pos_weight
    if cfg.pos_weight != "auto_sqrt":
        raise ValueError(f"bad pos_weight {cfg.pos_weight!r}")
    water = dry = 0
    for i in range(len(train_ds)):
        t = train_ds._cache[i][1] if train_ds._cache else train_ds[i]["mask"].numpy()
        water += int((t == 1).sum())
        dry += int((t == 0).sum())
    return float(np.sqrt(dry / max(1, water)))


def run(cfg: TrainConfig) -> dict[str, Any]:
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    seed_everything(cfg.seed)
    device = select_device(cfg.device)
    run_name = f"{cfg.name}-{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir = resolve(cfg.output_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    t_load = time.perf_counter()
    ds, stats, split_sha = build_datasets(cfg)
    load_s = time.perf_counter() - t_load
    pos_weight = resolve_pos_weight(cfg, ds["train"])
    g = torch.Generator().manual_seed(cfg.seed)
    train_dl = DataLoader(
        ds["train"],
        cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        generator=g,
        drop_last=len(ds["train"]) > cfg.batch_size,
    )
    eval_dl = {
        k: DataLoader(v, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
        for k, v in ds.items()
        if k != "train"
    }

    if cfg.arch == "unet":
        model = build_model(
            "unet", base_channels=cfg.base_channels, depth=cfg.depth, dropout=cfg.dropout
        )
    else:
        model = build_model(cfg.arch, pretrained=cfg.pretrained, dropout=cfg.dropout)
    model = model.to(device)
    loss_fn = BCEDiceLoss(cfg.bce_weight, cfg.dice_weight, pos_weight, IGNORE_INDEX).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = (
        len(train_dl)
        if cfg.max_train_batches is None
        else min(len(train_dl), cfg.max_train_batches)
    )
    sched = cosine_with_warmup(opt, cfg.warmup_epochs * steps, cfg.epochs * steps)

    meta = {
        "run_name": run_name,
        "device": device.type,
        "params": count_parameters(model),
        "pos_weight": pos_weight,
        "split_sha256": split_sha,
        "norm_stats": stats.__dict__,
        "git": git_state(),
        "torch": torch.__version__,
        "n_chips": {k: len(v) for k, v in ds.items()},
        "preload_seconds": round(load_s, 1),
    }
    print(
        json.dumps({k: meta[k] for k in ("run_name", "device", "params", "pos_weight", "n_chips")})
    )

    mlf = None
    if cfg.mlflow:
        import mlflow

        uri = cfg.mlflow_uri
        if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
            db = resolve(uri.removeprefix("sqlite:///"))
            db.parent.mkdir(parents=True, exist_ok=True)
            uri = f"sqlite:///{db}"
        mlflow.set_tracking_uri(uri)
        if mlflow.get_experiment_by_name(cfg.mlflow_experiment) is None:
            mlflow.create_experiment(
                cfg.mlflow_experiment,
                artifact_location=(resolve("data/mlflow/artifacts")).as_uri(),
            )
        mlflow.set_experiment(cfg.mlflow_experiment)
        mlf = mlflow.start_run(run_name=run_name)
        mlflow.log_params(
            {
                **cfg.to_dict(),
                "params": meta["params"],
                "pos_weight": pos_weight,
                "split_sha256": split_sha,
                "git_commit": meta["git"]["commit"],
                "git_dirty": meta["git"]["dirty"],
                "device": device.type,
            }
        )

    history: list[dict[str, Any]] = []
    best_iou, best_epoch, bad = -1.0, -1, 0
    ckpt = out_dir / "best.pt"
    try:
        for epoch in range(1, cfg.epochs + 1):
            tr = train_one_epoch(
                model,
                train_dl,
                loss_fn,
                opt,
                sched,
                device,
                amp=cfg.amp,
                grad_clip=cfg.grad_clip,
                max_batches=cfg.max_train_batches,
            )
            val, _ = evaluate(
                model, eval_dl["val"], device, loss_fn=loss_fn, threshold=cfg.threshold
            )
            row = {
                "epoch": epoch,
                "train_loss": tr["loss"],
                "val_loss": val["loss"],
                "val_iou": val["iou"],
                "val_f1": val["f1"],
                "lr": sched.get_last_lr()[0],
                "epoch_seconds": tr["epoch_seconds"],
            }
            history.append(row)
            improved = val["iou"] > best_iou + cfg.min_delta
            if improved:
                best_iou, best_epoch, bad = val["iou"], epoch, 0
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": cfg.to_dict(),
                        "meta": meta,
                        "epoch": epoch,
                        "val_iou": best_iou,
                    },
                    ckpt,
                )
            else:
                bad += 1
            print(
                f"epoch {epoch:3d} loss {tr['loss']:.4f} val_loss {val['loss']:.4f} "
                f"val_iou {val['iou']:.4f}{' *' if improved else ''} ({tr['epoch_seconds']:.1f}s)",
                flush=True,
            )
            if mlf:
                import mlflow

                mlflow.log_metrics({k: v for k, v in row.items() if k != "epoch"}, step=epoch)
            if bad >= cfg.patience:
                print(f"early stop at epoch {epoch} (best {best_epoch})")
                break

        # ---------------------------------------------------------------- final evaluation
        state = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        results: dict[str, Any] = {
            "meta": meta,
            "best_epoch": best_epoch,
            "epochs_run": len(history),
            "config": cfg.to_dict(),
            "splits": {},
        }
        grid = [round(0.05 * i, 2) for i in range(1, 20)]
        val_sweep = sweep_thresholds(model, eval_dl["val"], device, grid)
        tuned = max(grid, key=lambda t: val_sweep[t]["iou"])
        results["threshold"] = {
            "default": cfg.threshold,
            "val_tuned": tuned,
            "val_sweep_iou": {str(t): v["iou"] for t, v in val_sweep.items()},
        }
        results["splits_tuned_threshold"] = {}
        for name, dl in eval_dl.items():
            mt, _ = evaluate(model, dl, device, threshold=tuned)
            results["splits_tuned_threshold"][name] = mt
        for name, dl in eval_dl.items():
            m, acc = evaluate(model, dl, device, threshold=cfg.threshold, keep_per_chip=True)
            m["iou_ci95"] = bootstrap_iou_ci(
                [(v["tp"], v["fp"], v["fn"]) for v in acc.per_chip.values()]
            )
            results["splits"][name] = {
                "pooled": m,
                "per_event": group_scores(acc.per_chip),
                "per_chip": acc.per_chip,
            }
        results["latency"] = {
            f"bs{b}": benchmark_latency(model, device, (b, 2, cfg.latency_size, cfg.latency_size))
            for b in (1, 8)
        }
        (out_dir / "history.json").write_text(json.dumps(history, indent=1))
        (out_dir / "results.json").write_text(json.dumps(results, indent=1, default=float))
        if mlf:
            import mlflow

            for name, r in results["splits"].items():
                p = r["pooled"]
                mlflow.log_metrics(
                    {
                        f"{name}_{k}": v
                        for k, v in p.items()
                        if isinstance(v, float) and not np.isnan(v)
                    }
                )
            mlflow.log_metrics(
                {
                    "latency_ms_bs1": results["latency"]["bs1"]["latency_ms_mean"],
                    "best_epoch": best_epoch,
                }
            )
            mlflow.log_artifact(str(out_dir / "results.json"))
            mlflow.log_artifact(str(out_dir / "history.json"))
            mlflow.set_tag("checkpoint", str(ckpt))
    finally:
        if mlf:
            import mlflow

            mlflow.end_run()

    summary = {
        n: {k: round(r["pooled"][k], 4) for k in ("iou", "f1", "precision", "recall")}
        for n, r in results["splits"].items()
    }
    print(json.dumps({"run": run_name, "best_epoch": best_epoch, "results": summary}, indent=1))
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--max-train-batches", type=int)
    ap.add_argument("--device")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--no-mlflow", action="store_true")
    a = ap.parse_args()
    cfg = TrainConfig.from_yaml(
        a.config,
        epochs=a.epochs,
        max_train_batches=a.max_train_batches,
        device=a.device,
        seed=a.seed,
    )
    if a.no_mlflow:
        cfg.mlflow = False
    run(cfg)


if __name__ == "__main__":
    main()
