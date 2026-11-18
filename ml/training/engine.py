"""Training / evaluation loops (device-agnostic: cuda, mps, cpu)."""

from __future__ import annotations

import math
import random
import time
from collections.abc import Iterable

import numpy as np
import torch
from torch import nn

from ml.evaluation.metrics import MetricAccumulator


def select_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def cosine_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    def f(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def _autocast(device: torch.device, enabled: bool):
    if not enabled or device.type == "cpu":
        return torch.autocast(device_type="cpu", enabled=False)
    return torch.autocast(device_type=device.type, dtype=torch.float16)


def train_one_epoch(
    model: nn.Module,
    loader: Iterable,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    *,
    amp: bool = False,
    grad_clip: float | None = None,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.train()
    total, n = 0.0, 0
    t0 = time.perf_counter()
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = batch["image"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, amp):
            logits = model(x)
        loss = loss_fn(logits.float(), y)
        loss.backward()
        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        total += loss.item() * x.shape[0]
        n += x.shape[0]
    sync(device)
    return {"loss": total / max(1, n), "epoch_seconds": time.perf_counter() - t0}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: Iterable,
    device: torch.device,
    *,
    loss_fn: nn.Module | None = None,
    threshold: float = 0.5,
    keep_per_chip: bool = False,
) -> tuple[dict[str, float], MetricAccumulator]:
    model.eval()
    acc = MetricAccumulator()
    total, n = 0.0, 0
    for batch in loader:
        x = batch["image"].to(device)
        y = batch["mask"].to(device)
        logits = model(x)
        if loss_fn is not None:
            total += loss_fn(logits, y).item() * x.shape[0]
            n += x.shape[0]
        pred = (torch.sigmoid(logits[:, 0]) > threshold).long()
        acc.update(pred, y, chips=list(batch["chip"]) if keep_per_chip else None)
    metrics = acc.compute()
    if loss_fn is not None:
        metrics["loss"] = total / max(1, n)
    return metrics, acc


@torch.no_grad()
def benchmark_latency(
    model: nn.Module,
    device: torch.device,
    shape: tuple[int, int, int, int] = (1, 2, 512, 512),
    warmup: int = 5,
    iters: int = 30,
) -> dict[str, float]:
    """Forward-pass latency in ms (synchronised) on random input of ``shape``."""
    model.eval()
    x = torch.randn(*shape, device=device)
    for _ in range(warmup):
        model(x)
    sync(device)
    times = []
    for _ in range(iters):
        t = time.perf_counter()
        model(x)
        sync(device)
        times.append((time.perf_counter() - t) * 1000)
    arr = np.array(times)
    return {
        "latency_ms_mean": float(arr.mean()),
        "latency_ms_p50": float(np.percentile(arr, 50)),
        "latency_ms_p95": float(np.percentile(arr, 95)),
        "batch": shape[0],
        "device": device.type,
    }


@torch.no_grad()
def predict_probs(model: nn.Module, loader: Iterable, device: torch.device):
    """Yield (probs (B,H,W) cpu float, target (B,H,W) cpu long, chips) batches."""
    model.eval()
    for batch in loader:
        p = torch.sigmoid(model(batch["image"].to(device))[:, 0]).float().cpu()
        yield p, batch["mask"], list(batch["chip"])


def sweep_thresholds(
    model: nn.Module, loader: Iterable, device: torch.device, thresholds: Iterable[float]
) -> dict[float, dict[str, float]]:
    """Pooled scores at each threshold from a single forward pass over ``loader``."""
    from ml.evaluation.metrics import IGNORE_INDEX, scores

    ts = [round(float(t), 3) for t in thresholds]
    counts = {t: [0, 0, 0, 0] for t in ts}
    for p, y, _ in predict_probs(model, loader, device):
        valid = y != IGNORE_INDEX
        pos = (y == 1) & valid
        for t in ts:
            pred = (p > t) & valid
            c = counts[t]
            c[0] += int((pred & pos).sum())
            c[1] += int((pred & ~pos & valid).sum())
            c[2] += int((~pred & pos).sum())
            c[3] += int((~pred & ~pos & valid).sum())
    return {t: scores(*c) for t, c in counts.items()}
