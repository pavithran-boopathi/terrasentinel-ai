#!/usr/bin/env python3
"""Render markdown tables from results.json files (baselines and U-Net runs).

    python scripts/summarize_experiments.py --baselines data/checkpoints/baselines-*/results.json \
        --runs data/checkpoints/unet-*/results.json

Prints markdown to stdout; copy into EXPERIMENTS.md. Numbers are never typed by hand.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def f(x: float) -> str:
    return "—" if x is None or x != x else f"{x:.3f}"


def ci(p: dict) -> str:
    lo, hi = p.get("iou_ci95", (float("nan"), float("nan")))
    return f"{f(p['iou'])} [{f(lo)}, {f(hi)}]"


HEADER = "| Method | Split | IoU [95% CI] | F1 | Precision | Recall | Chips |\n|---|---|---|---|---|---|---|"


def baseline_rows(path: Path) -> dict[str, list[str]]:
    data = json.loads(path.read_text())
    out: dict[str, list[str]] = {}
    for scheme, r in data.items():
        rows = []
        for m, per_split in r["methods"].items():
            for s, v in per_split.items():
                p = v["pooled"]
                rows.append(
                    f"| {m} | {s} | {ci(p)} | {f(p['f1'])} | {f(p['precision'])} | "
                    f"{f(p['recall'])} | {len(v['per_chip'])} |"
                )
        out[scheme] = rows
    return out


def run_rows(path: Path) -> tuple[str, list[str], dict]:
    r = json.loads(path.read_text())
    scheme = r["config"]["split_scheme"]
    rows = []
    for s, v in r["splits"].items():
        p = v["pooled"]
        rows.append(
            f"| U-Net ({r['meta']['run_name']}) | {s} | {ci(p)} | {f(p['f1'])} | "
            f"{f(p['precision'])} | {f(p['recall'])} | {len(v['per_chip'])} |"
        )
    return scheme, rows, r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", type=Path, nargs="*", default=[])
    ap.add_argument("--runs", type=Path, nargs="*", default=[])
    a = ap.parse_args()
    tables: dict[str, list[str]] = {}
    for b in a.baselines:
        for scheme, rows in baseline_rows(b).items():
            tables.setdefault(scheme, []).extend(rows)
    runs = []
    for rp in a.runs:
        scheme, rows, r = run_rows(rp)
        tables.setdefault(scheme, []).extend(rows)
        runs.append(r)
    for scheme, rows in tables.items():
        print(f"\n#### `{scheme}`\n\n{HEADER}")
        print("\n".join(rows))
    by_cfg: dict[tuple[str, str], list[float]] = {}
    for r in runs:
        for s, v in r["splits"].items():
            by_cfg.setdefault((r["config"]["name"], s), []).append(v["pooled"]["iou"])
    if any(len(v) > 1 for v in by_cfg.values()):
        print(
            "\n#### Across seeds (pooled flood IoU)\n\n| Config | Split | Seeds | Mean | Std | Min | Max |"
        )
        print("|---|---|---|---|---|---|---|")
        for (name, s), vals in sorted(by_cfg.items()):
            a = np.array(vals)
            print(
                f"| {name} | {s} | {len(a)} | {a.mean():.3f} | {a.std(ddof=1) if len(a) > 1 else 0:.3f} "
                f"| {a.min():.3f} | {a.max():.3f} |"
            )
    for r in runs:
        m = r["meta"]
        lat = r["latency"]
        print(
            f"\n**{m['run_name']}** — commit `{m['git']['commit'][:8]}`"
            f"{' (dirty)' if m['git']['dirty'] else ''}, device {m['device']}, "
            f"{m['params']:,} params, pos_weight {m['pos_weight']:.2f}, best epoch "
            f"{r['best_epoch']}/{r['epochs_run']}, split sha `{m['split_sha256'][:12]}`. "
            f"Latency 512² ({lat['bs1']['device']}): bs1 {lat['bs1']['latency_ms_mean']:.1f} ms "
            f"(p95 {lat['bs1']['latency_ms_p95']:.1f}), bs8 {lat['bs8']['latency_ms_mean']:.1f} ms "
            f"→ {lat['bs8']['latency_ms_mean'] / 8:.1f} ms/chip."
        )
        for s, v in r["splits"].items():
            ev = v["per_event"]
            if len(ev) > 1:
                print(
                    f"\nPer-event IoU on `{s}`: "
                    + ", ".join(f"{e} {f(x['iou'])} (n={x['chips']})" for e, x in ev.items())
                )


if __name__ == "__main__":
    main()
