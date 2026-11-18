#!/usr/bin/env python3
"""Validate the downloaded Sen1Floods11 hand-labeled data, build reproducible splits and
normalisation statistics, and write a data report.

    python scripts/prepare_sen1floods11.py [--raw data/raw/sen1floods11] [--out data/processed/sen1floods11]

Outputs
  <out>/splits/{official,geo_holdout_v1}.json   chip lists + sha256
  <out>/splits/loeo/<event>.json                leave-one-event-out folds
  <out>/norm_stats_{official,geo_holdout_v1}.json   computed on each scheme's TRAIN split only
  <out>/validation.json                         per-file issues
  <out>/quicklook/*.png                         SAR false colour + label overlays (not committed)
  docs/data/sen1floods11_report.md              aggregate statistics (committed)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from geospatial.raster import check_alignment, validate_raster  # noqa: E402
from geospatial.viz import nodata_hatch, overlay_mask, sar_false_color  # noqa: E402
from ml.data.sen1floods11 import (  # noqa: E402
    DB_CLIP,
    GEO_HOLDOUT_V1,
    HAND,
    IGNORE_INDEX,
    Chip,
    check_no_leakage,
    compute_norm_stats,
    discover_chips,
    event_holdout_splits,
    label_summary,
    leave_one_event_out,
    official_splits,
    read_chip,
    splits_to_json,
)

EXPECTED_CHIPS = 446


def classify(issues: dict[str, list[str]]) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Split issues into fatal (chip excluded) and warnings (handled downstream, e.g. dB clipping)."""
    fatal, warn = {}, {}
    for name, msgs in issues.items():
        f = [m for m in msgs if "outside" not in m]
        w = [m for m in msgs if "outside" in m]
        if f:
            fatal[name] = f
        if w:
            warn[name] = w
    return fatal, warn


def drop(splits: dict[str, list[Chip]], excluded: set[str]) -> dict[str, list[Chip]]:
    return {k: [c for c in v if c.name not in excluded] for k, v in splits.items()}


def validate_all(chips: list[Chip], raw: Path) -> dict[str, list[str]]:
    issues: dict[str, list[str]] = {}
    for c in chips:
        found: list[str] = []
        s1 = validate_raster(
            c.s1_path(raw),
            expected_count=2,
            expected_dtype="float32",
            expected_shape=(512, 512),
            expected_crs="EPSG:4326",
            value_range=(-100.0, 40.0),
        )
        lab = validate_raster(
            c.label_path(raw),
            expected_count=1,
            expected_dtype="int16",
            expected_shape=(512, 512),
            expected_crs="EPSG:4326",
            allowed_values={-1, 0, 1},
        )
        found += [f"S1: {e}" for e in s1.errors] + [f"label: {e}" for e in lab.errors]
        found += [f"S1/label: {p}" for p in check_alignment(c.s1_path(raw), c.label_path(raw))]
        jrc = raw / HAND / "JRCWaterHand" / f"{c.name}_JRCWaterHand.tif"
        if jrc.exists():
            found += [f"S1/JRC: {p}" for p in check_alignment(c.s1_path(raw), jrc)]
        else:
            found.append("JRC file missing")
        if found:
            issues[c.name] = found
    return issues


def per_event_table(chips: list[Chip], raw: Path) -> list[dict]:
    agg: dict[str, dict] = defaultdict(lambda: defaultdict(float))
    vv_samples, vh_samples = defaultdict(list), defaultdict(list)
    rng = np.random.default_rng(0)
    for c in chips:
        img, t = read_chip(c, raw)
        a = agg[c.event]
        a["chips"] += 1
        a["water"] += (t == 1).sum()
        a["dry"] += (t == 0).sum()
        a["ignore"] += (t == IGNORE_INDEX).sum()
        a["s1_nan"] += (~np.isfinite(img).all(0)).sum()
        a["px"] += t.size
        valid = np.isfinite(img).all(0)
        idx = rng.choice(valid.sum(), size=min(2000, int(valid.sum())), replace=False)
        vv_samples[c.event].append(img[0][valid][idx])
        vh_samples[c.event].append(img[1][valid][idx])
    rows = []
    for ev in sorted(agg):
        a = agg[ev]
        vv = np.concatenate(vv_samples[ev])
        vh = np.concatenate(vh_samples[ev])
        rows.append(
            {
                "event": ev,
                "chips": int(a["chips"]),
                "water_pct_of_labelled": 100 * a["water"] / max(1, a["water"] + a["dry"]),
                "ignore_pct": 100 * a["ignore"] / a["px"],
                "s1_nan_pct": 100 * a["s1_nan"] / a["px"],
                "vv_p1_p50_p99": np.percentile(vv, [1, 50, 99]).round(1).tolist(),
                "vh_p1_p50_p99": np.percentile(vh, [1, 50, 99]).round(1).tolist(),
            }
        )
    return rows


def quicklooks(chips: list[Chip], raw: Path, out: Path, n_per_event: int = 1) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    by_event: dict[str, list[Chip]] = defaultdict(list)
    for c in chips:
        by_event[c.event].append(c)
    picks = [by_event[e][len(by_event[e]) // 2] for e in sorted(by_event)][: 12 * n_per_event]
    fig, axes = plt.subplots(len(picks), 2, figsize=(6, 3 * len(picks)))
    for row, c in zip(axes, picks, strict=True):
        img, t = read_chip(c, raw)
        rgb = sar_false_color(img[0], img[1])
        row[0].imshow(rgb)
        row[0].set_title(f"{c.name} S1 (VV,VH,ratio)", fontsize=7)
        lab = np.where(t == IGNORE_INDEX, -1, t.astype(np.int16))
        row[1].imshow(nodata_hatch(overlay_mask(rgb, lab), t == IGNORE_INDEX))
        row[1].set_title("hand label: blue=water, magenta=ignore", fontsize=7)
        for ax in row:
            ax.axis("off")
    fig.tight_layout()
    path = out / "events_overview.png"
    fig.savefig(path, dpi=80)
    plt.close(fig)
    return [path]


def fmt_summary(s: dict) -> str:
    return (
        f"{s['water_fraction_of_labelled'] * 100:.1f}% water of labelled px · "
        f"{s['ignore_fraction'] * 100:.1f}% ignored"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=REPO / "data/raw/sen1floods11")
    ap.add_argument("--out", type=Path, default=REPO / "data/processed/sen1floods11")
    ap.add_argument("--report", type=Path, default=REPO / "docs/data/sen1floods11_report.md")
    args = ap.parse_args()
    t0 = time.time()

    chips = discover_chips(args.raw)
    print(f"chips with S1 + label: {len(chips)}")
    if len(chips) != EXPECTED_CHIPS:
        print(f"WARNING: expected {EXPECTED_CHIPS}")

    print("validating rasters ...")
    issues = validate_all(chips, args.raw)
    (args.out).mkdir(parents=True, exist_ok=True)
    (args.out / "validation.json").write_text(json.dumps(issues, indent=1))
    fatal, warn = classify(issues)
    excluded = set(fatal)
    print(f"  chips excluded: {sorted(excluded)}; warnings: {sorted(warn)}")
    official_raw = official_splits(args.raw)
    excluded_from = {
        n: [k for k, v in official_raw.items() if any(c.name == n for c in v)] for n in excluded
    }
    usable = [c for c in chips if c.name not in excluded]

    schemes = {
        "official": drop(official_raw, excluded),
        "geo_holdout_v1": event_holdout_splits(
            usable, **{"test_events": GEO_HOLDOUT_V1["test"], "val_events": GEO_HOLDOUT_V1["val"]}
        ),
    }
    split_dir = args.out / "splits"
    (split_dir / "loeo").mkdir(parents=True, exist_ok=True)
    digests, stats, summaries, leakage = {}, {}, {}, {}
    for name, sp in schemes.items():
        missing = {c for v in sp.values() for c in v} - set(chips)
        if missing:
            raise SystemExit(f"{name}: {len(missing)} split chips not on disk")
        text, digests[name] = splits_to_json(
            sp, {"name": name, "dataset": "sen1floods11-v1.1-hand", "excluded": fatal}
        )
        (split_dir / f"{name}.json").write_text(text)
        leakage[name] = {
            "chip_overlap": check_no_leakage(sp, by_event=False),
            "event_overlap": check_no_leakage(sp, by_event=True),
        }
        stats[name] = compute_norm_stats(sp["train"], args.raw, source=f"{name}/train")
        (args.out / f"norm_stats_{name}.json").write_text(stats[name].to_json())
        summaries[name] = {k: (len(v), label_summary(v, args.raw)) for k, v in sp.items()}
        print(f"  {name}: " + ", ".join(f"{k}={len(v)}" for k, v in sp.items()))
    for fold in leave_one_event_out(usable):
        fsp = {"train": fold["train"], "test": fold["test"]}
        text, _ = splits_to_json(fsp, {"name": f"loeo/{fold['held_out']}", "excluded": fatal})
        (split_dir / "loeo" / f"{fold['held_out']}.json").write_text(text)

    events = per_event_table(usable, args.raw)
    looks = quicklooks(usable, args.raw, args.out / "quicklook")
    with rasterio.open(chips[0].s1_path(args.raw)) as s:
        res_deg = s.res

    # ------------------------------------------------------------------ report
    L = [
        "# Sen1Floods11 (v1.1, hand-labeled) — data report",
        "",
        f"Generated by `scripts/prepare_sen1floods11.py` on {time.strftime('%Y-%m-%d')} from the "
        f"files listed in `data/raw/sen1floods11/MANIFEST.json`. All numbers below are computed, "
        f"not copied from the paper.",
        "",
        "## Source",
        "- Bucket `gs://sen1floods11/v1.1` (public), downloaded over HTTPS with per-file MD5 checks",
        "- Bonafilia et al., *Sen1Floods11*, CVPR Workshops 2020",
        "- **License:** none declared in the GitHub repo, bucket, or STAC catalog. Sentinel-1 "
        "imagery is Copernicus open data. Used here for research with citation; raw data and "
        "label imagery are not redistributed in this repository.",
        "",
        "## Structure",
        f"- Chips with S1 + hand label: **{len(chips)}** (expected {EXPECTED_CHIPS})",
        f"- 512×512 px, EPSG:4326, pixel {res_deg[0]:.3e}° (≈10 m), S1 bands VV, VH in dB (float32)",
        "- Labels: -1 no data, 0 not water, 1 water → encoded as 0/1 with "
        f"ignore index {IGNORE_INDEX} (also applied where any S1 band is NaN)",
        f"- Raster validation (count, dtype, shape, CRS, value range, label values, S1↔label↔JRC "
        f"grid alignment): **{len(fatal)} chip(s) excluded**, {len(warn)} warning(s)",
        *[
            f"  - Excluded `{n}` (official split: {', '.join(excluded_from[n]) or '—'}): "
            f"{'; '.join(m)}"
            for n, m in sorted(fatal.items())
        ],
        *[
            f"  - Warning `{n}`: {'; '.join(m)} — handled by clipping to {DB_CLIP} dB"
            for n, m in sorted(warn.items())
        ],
        "- Exclusions are applied to every split scheme and recorded in each split file's metadata."
        " Results on the official test split are therefore on "
        "{n_official_test} chips (official list: 90).",
        "",
        "## Per event",
        "",
        "| Event | Chips | Water % (labelled) | Ignored % | S1 NaN % | VV dB p1/p50/p99 "
        "| VH dB p1/p50/p99 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in events:
        L.append(
            f"| {r['event']} | {r['chips']} | {r['water_pct_of_labelled']:.1f} | "
            f"{r['ignore_pct']:.1f} | {r['s1_nan_pct']:.2f} | "
            f"{' / '.join(map(str, r['vv_p1_p50_p99']))} | {' / '.join(map(str, r['vh_p1_p50_p99']))} |"
        )
    L += ["", "## Split schemes", ""]
    for name, sp in schemes.items():
        L += [f"### `{name}` (sha256 `{digests[name][:12]}…`)", ""]
        for k, (n, s) in summaries[name].items():
            evs = sorted({c.event for c in sp[k]})
            L.append(f"- **{k}**: {n} chips · {fmt_summary(s)} · events: {', '.join(evs)}")
        lk = leakage[name]
        L.append(
            f"- Chip overlap between splits: {lk['chip_overlap'] or 'none'}; "
            f"event overlap: {lk['event_overlap'] or 'none'}"
        )
        st = stats[name]
        L += [
            f"- Normalisation (train only, clip {st.clip} dB, {st.n_pixels:,} px): "
            f"mean VV/VH = {st.mean[0]:.3f} / {st.mean[1]:.3f}, std = {st.std[0]:.3f} / {st.std[1]:.3f}",
            "",
        ]
    L += [
        "**Why two schemes.** The official train/val/test splits draw chips from the same flood "
        "events (only Bolivia is held out), so neighbouring, correlated chips can appear in both "
        "training and test. `geo_holdout_v1` holds out entire events "
        f"(test: {', '.join(GEO_HOLDOUT_V1['test'])}; val: {', '.join(GEO_HOLDOUT_V1['val'])}) "
        "to measure generalisation to unseen regions. Leave-one-event-out folds "
        f"({len({c.event for c in chips})}) are written to `splits/loeo/` for Phase 8.",
        "",
        "## Observations",
        "- Strong class imbalance: water is a minority class in every event → use Dice/Lovász or "
        "weighted BCE and report flood-class IoU, not pixel accuracy.",
        "- Large ignored fractions in some events (pixels annotators marked as not valid) "
        "→ metrics must exclude ignore pixels.",
        "- S1 NaNs (pixels with no radar coverage) exist and are masked from loss and metrics.",
        "",
        f"Quick-look figure (not committed): `{looks[0].relative_to(REPO)}`",
    ]
    text = "\n".join(L).replace("{n_official_test}", str(len(schemes["official"]["test"])))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(text + "\n")
    print(f"report → {args.report.relative_to(REPO)}  ({time.time() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
