"""Train the infrastructure + flood-attribution model on SpaceNet 8 (pre+post RGB → 5 classes).

    python -m ml.training.train_spacenet8 [--epochs 60 --seed 42]

Model: U-Net, ImageNet ResNet-34 encoder, 6 input channels (pre RGB ‖ post RGB warped to the pre
grid). Loss: class-weighted cross-entropy (weights ∝ 1/sqrt(freq), train split only), ignore 255.
Reports per-class pixel IoU and object-level flood attribution (per building / along road
centrelines) on val and the held-out spatial test block, plus two reference baselines.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ml.data.spacenet8 import CLASSES, SN8Dataset, object_level_scores, summarize_object_scores
from ml.training.engine import cosine_with_warmup, seed_everything, select_device
from ml.training.train import git_state

REPO = Path(__file__).resolve().parents[2]
PROC = REPO / "data/processed/spacenet8"


def build(pretrained: bool = True) -> nn.Module:
    import segmentation_models_pytorch as smp

    return smp.Unet(
        "resnet34",
        encoder_weights="imagenet" if pretrained else None,
        in_channels=6,
        classes=len(CLASSES),
    )


class CEDiceLoss(nn.Module):
    """Weighted cross-entropy + soft Dice over the foreground classes (1..C-1), ignore 255.
    Dice counters the foreground over-prediction that a re-weighted CE alone produces."""

    def __init__(self, weights: torch.Tensor, dice_weight: float = 1.0) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weights, ignore_index=255)
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = self.ce(logits, y)
        if not self.dice_weight:
            return loss
        valid = (y != 255).unsqueeze(1)
        p = torch.softmax(logits, 1) * valid
        onehot = nn.functional.one_hot(y.clamp(max=logits.shape[1] - 1), logits.shape[1])
        onehot = onehot.permute(0, 3, 1, 2).float() * valid
        dims = (0, 2, 3)
        inter = (p * onehot).sum(dims)[1:]
        denom = (p.sum(dims) + onehot.sum(dims))[1:]
        dice = (2 * inter + 1) / (denom + 1)
        return loss + self.dice_weight * (1 - dice.mean())


def class_weights(ds: SN8Dataset) -> torch.Tensor:
    counts = np.zeros(len(CLASSES))
    for n in ds.names:
        counts += np.bincount(ds.load(n)["mask"].ravel(), minlength=len(CLASSES))[: len(CLASSES)]
    w = 1 / np.sqrt(counts / counts.sum())
    return torch.tensor(w / w.mean(), dtype=torch.float32)


@torch.no_grad()
def predict_tile(model: nn.Module, ds: SN8Dataset, name: str, device) -> np.ndarray:
    d = ds.load(name)
    x = SN8Dataset.to_input(d["pre"], d["post"])
    h, w = x.shape[1:]
    x = np.pad(x, ((0, 0), (0, (-h) % 32), (0, (-w) % 32)))
    logits = model(torch.from_numpy(x)[None].to(device))[0, :, :h, :w]
    return logits.argmax(0).cpu().numpy().astype(np.uint8)


def evaluate_split(model, ds: SN8Dataset, device, predictor=None) -> dict[str, Any]:
    """``predictor(name) -> class map`` overrides the model (used for baselines)."""
    model.eval() if model is not None else None
    conf = np.zeros((len(CLASSES), len(CLASSES)), np.int64)
    obj: dict[str, int] = {}
    for name in ds.names:
        d = ds.load(name)
        pred = predictor(name) if predictor else predict_tile(model, ds, name, device)
        y = d["mask"]
        conf += np.bincount(
            len(CLASSES) * y.ravel().astype(np.int64) + pred.ravel(), minlength=len(CLASSES) ** 2
        ).reshape(len(CLASSES), len(CLASSES))
        for k, v in object_level_scores(
            pred, d["buildings"], d["building_flooded"], d["road_center"]
        ).items():
            obj[k] = obj.get(k, 0) + v
    tp = np.diag(conf)
    iou = tp / np.maximum(conf.sum(0) + conf.sum(1) - tp, 1)
    per_class = {c: float(v) for c, v in zip(CLASSES, iou, strict=True)}
    return {
        "pixel_iou": per_class,
        "mean_iou_fg": float(iou[1:].mean()),
        "object": summarize_object_scores(obj),
        "object_counts": obj,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--dice-weight", type=float, default=1.0)
    ap.add_argument("--focus-prob", type=float, default=0.5)
    ap.add_argument("--no-mlflow", action="store_true")
    a = ap.parse_args()
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    seed_everything(a.seed)
    device = select_device()
    split = json.loads((PROC / "split.json").read_text())["splits"]
    tiles = PROC / "tiles"
    train = SN8Dataset(
        split["train"],
        tiles,
        crop=a.crop,
        augment=True,
        seed=a.seed,
        focus_classes=(2, 4),
        focus_prob=a.focus_prob,
    )
    val = SN8Dataset(split["val"], tiles)
    test = SN8Dataset(split["test"], tiles)
    for d in (train, val, test):
        for n in d.names:
            d.load(n)
    weights = class_weights(train)
    model = build().to(device)
    loss_fn = CEDiceLoss(weights.to(device), a.dice_weight)
    # 4 random crops per tile per epoch
    dl = DataLoader(
        torch.utils.data.ConcatDataset([train] * 4),
        a.batch,
        shuffle=True,
        generator=torch.Generator().manual_seed(a.seed),
        drop_last=True,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = cosine_with_warmup(opt, 2 * len(dl), a.epochs * len(dl))
    run = f"sn8-unet-resnet34-{time.strftime('%Y%m%d-%H%M%S')}"
    out = REPO / "data/checkpoints" / run
    out.mkdir(parents=True)
    meta = {
        "run_name": run,
        "device": device.type,
        "class_weights": weights.tolist(),
        "git": git_state(),
        "seed": a.seed,
        "args": vars(a),
        "n_tiles": {k: len(v) for k, v in split.items()},
    }
    mlf = None
    if not a.no_mlflow:
        import mlflow

        mlflow.set_tracking_uri(f"sqlite:///{REPO / 'data/mlflow/mlflow.db'}")
        mlflow.set_experiment("spacenet8-infrastructure")
        mlf = mlflow.start_run(run_name=run)
        mlflow.log_params({**vars(a), "git_commit": meta["git"]["commit"]})
    best, best_ep, bad, hist = -1.0, 0, 0, []
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, tot = time.perf_counter(), 0.0
        for b in dl:
            x, y = b["image"].to(device), b["mask"].to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
        v = evaluate_split(model, val, device)
        score = v["mean_iou_fg"]
        row = {
            "epoch": ep,
            "loss": tot / len(dl),
            "val_mean_iou_fg": score,
            "val_flooded_building_f1": v["object"]["flooded_building"]["f1"],
            "seconds": time.perf_counter() - t0,
        }
        hist.append(row)
        improved = score > best + 1e-3
        if improved:
            best, best_ep, bad = score, ep, 0
            torch.save({"model": model.state_dict(), "meta": meta, "epoch": ep}, out / "best.pt")
        else:
            bad += 1
        print(
            f"epoch {ep:3d} loss {row['loss']:.4f} val mIoU(fg) {score:.4f} "
            f"flooded-bld F1 {row['val_flooded_building_f1']:.3f}{' *' if improved else ''} "
            f"({row['seconds']:.0f}s)",
            flush=True,
        )
        if mlf:
            import mlflow

            mlflow.log_metrics({k: float(v) for k, v in row.items() if k != "epoch"}, step=ep)
        if bad >= a.patience:
            print(f"early stop at {ep} (best {best_ep})")
            break
    model.load_state_dict(
        torch.load(out / "best.pt", map_location=device, weights_only=False)["model"]
    )
    results: dict[str, Any] = {
        "meta": meta,
        "best_epoch": best_ep,
        "epochs_run": len(hist),
        "val": evaluate_split(model, val, device),
        "test": evaluate_split(model, test, device),
    }

    # Reference baselines on test: (1) all-dry: perfect geometry from labels, never flooded;
    # (2) all-flooded. They bound what flood *attribution* adds beyond knowing where assets are.
    def oracle(flood: bool):
        def f(name):
            m = test.load(name)["mask"].copy()
            m[(m == 1) | (m == 2)] = 2 if flood else 1
            m[(m == 3) | (m == 4)] = 4 if flood else 3
            return m

        return f

    results["baselines_test"] = {
        "oracle_geometry_all_dry": evaluate_split(None, test, device, oracle(False))["object"],
        "oracle_geometry_all_flooded": evaluate_split(None, test, device, oracle(True))["object"],
    }
    (out / "history.json").write_text(json.dumps(hist, indent=1))
    (out / "results.json").write_text(json.dumps(results, indent=1, default=float))
    if mlf:
        import mlflow

        t = results["test"]
        mlflow.log_metrics(
            {
                "test_mean_iou_fg": t["mean_iou_fg"],
                **{f"test_iou_{c}": v for c, v in t["pixel_iou"].items()},
                "test_flooded_building_f1": t["object"]["flooded_building"]["f1"],
                "test_flooded_road_f1": t["object"]["flooded_road_length"]["f1"],
            }
        )
        mlflow.log_artifact(str(out / "results.json"))
        mlflow.end_run()
    t = results["test"]
    print(
        json.dumps(
            {
                "run": run,
                "best_epoch": best_ep,
                "test_pixel_iou": t["pixel_iou"],
                "test_object": t["object"],
            },
            indent=1,
            default=float,
        )
    )


if __name__ == "__main__":
    main()
