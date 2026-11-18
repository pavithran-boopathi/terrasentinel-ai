"""Train the building-damage model on xBD (pre+post RGB → background + 4 damage classes).

    python -m ml.training.train_xbd --root data/raw/xbd --test-disasters hurricane-michael \
        --val-disasters palu-tsunami [--epochs 30]

Not trained in this repository yet: xBD requires registration (xview2.org). The pipeline is
exercised end to end on synthetic xBD-format data in tests/ml/test_xbd.py.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ml.data.xbd import CLASSES, IGNORE, XBDDataset, event_split, index_tiles, xview2_score
from ml.training.engine import cosine_with_warmup, seed_everything, select_device

REPO = Path(__file__).resolve().parents[2]


def build(pretrained: bool) -> nn.Module:
    import segmentation_models_pytorch as smp

    return smp.Unet(
        "resnet34",
        encoder_weights="imagenet" if pretrained else None,
        in_channels=6,
        classes=len(CLASSES),
    )


@torch.no_grad()
def confusion(model: nn.Module, ds: XBDDataset, device) -> np.ndarray:
    model.eval()
    conf = np.zeros((len(CLASSES), len(CLASSES)), np.int64)
    for i in range(len(ds)):
        b = ds[i]
        x = b["image"][None].to(device)
        h, w = x.shape[-2:]
        xp = torch.nn.functional.pad(x, (0, (-w) % 32, 0, (-h) % 32))
        pred = model(xp)[0, :, :h, :w].argmax(0).cpu().numpy()
        y = b["mask"].numpy()
        m = y != IGNORE
        conf += np.bincount(len(CLASSES) * y[m] + pred[m], minlength=len(CLASSES) ** 2).reshape(
            len(CLASSES), len(CLASSES)
        )
    return conf


def run(
    root: Path,
    test_d: set[str],
    val_d: set[str],
    *,
    epochs: int,
    crop: int,
    batch: int,
    lr: float,
    seed: int,
    pretrained: bool,
    out_root: Path,
    device: str = "auto",
) -> dict:
    seed_everything(seed)
    dev = select_device(device)
    split = event_split(index_tiles(root), test_d, val_d)
    if not split["train"]:
        raise SystemExit(f"no training tiles under {root}")
    train = XBDDataset(split["train"], crop=crop, seed=seed)
    val, test = XBDDataset(split["val"]), XBDDataset(split["test"])
    model = build(pretrained).to(dev)
    # Damage classes are rare relative to background and to "no-damage".
    weights = torch.tensor([0.1, 1.0, 3.0, 3.0, 2.0], device=dev)
    loss_fn = nn.CrossEntropyLoss(weight=weights, ignore_index=IGNORE)
    dl = DataLoader(train, batch, shuffle=True, generator=torch.Generator().manual_seed(seed))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = cosine_with_warmup(opt, len(dl), epochs * len(dl))
    out = out_root / f"xbd-unet-resnet34-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True)
    best, hist = -1.0, []
    for ep in range(1, epochs + 1):
        model.train()
        for b in dl:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(b["image"].to(dev)), b["mask"].to(dev))
            loss.backward()
            opt.step()
            sched.step()
        score = xview2_score(confusion(model, val, dev))["xview2_score"] if len(val) else 0.0
        hist.append({"epoch": ep, "loss": float(loss.item()), "val_xview2": score})
        if score >= best:
            best = score
            torch.save({"model": model.state_dict(), "epoch": ep}, out / "best.pt")
    model.load_state_dict(torch.load(out / "best.pt", map_location=dev)["model"])
    res = {
        "splits": {k: [f"{t.disaster}_{t.tile_id}" for t in v] for k, v in split.items()},
        "history": hist,
        "test": xview2_score(confusion(model, test, dev)) if len(test) else None,
        "val": xview2_score(confusion(model, val, dev)) if len(val) else None,
    }
    (out / "results.json").write_text(json.dumps(res, indent=1, default=float))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=REPO / "data/raw/xbd")
    ap.add_argument("--test-disasters", nargs="+", required=True)
    ap.add_argument("--val-disasters", nargs="+", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    res = run(
        a.root,
        set(a.test_disasters),
        set(a.val_disasters),
        epochs=a.epochs,
        crop=a.crop,
        batch=a.batch,
        lr=a.lr,
        seed=a.seed,
        pretrained=True,
        out_root=REPO / "data/checkpoints",
    )
    print(json.dumps({"test": res["test"], "val": res["val"]}, indent=1, default=float))


if __name__ == "__main__":
    main()
