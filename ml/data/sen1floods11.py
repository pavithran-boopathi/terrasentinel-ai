"""Sen1Floods11 (v1.1, hand-labeled) — chip indexing, reproducible splits, normalisation, Dataset.

Layout (as downloaded by ``scripts/download_sen1floods11.py``)::

    <root>/data/flood_events/HandLabeled/S1Hand/<Event>_<id>_S1Hand.tif
        2×512×512 float32 dB (VV, VH), NaN = nodata
    <root>/data/flood_events/HandLabeled/LabelHand/<Event>_<id>_LabelHand.tif
        1×512×512 int16 (-1 nodata, 0 dry, 1 water)
    <root>/splits/flood_handlabeled/flood_{train,valid,test,bolivia}_data.csv

Label convention used by this project: 0 = not water, 1 = water, ``IGNORE_INDEX`` = no data
(either the hand label is -1 or any S1 band is NaN at that pixel).
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

IGNORE_INDEX = 255
BANDS = ("VV", "VH")
# Clip range in dB before normalisation; values outside are physically implausible / artefacts.
DB_CLIP = (-50.0, 5.0)

HAND = Path("data/flood_events/HandLabeled")
SPLIT_DIR = Path("splits/flood_handlabeled")
OFFICIAL_SPLITS = {
    "train": "flood_train_data.csv",
    "val": "flood_valid_data.csv",
    "test": "flood_test_data.csv",
    "bolivia": "flood_bolivia_data.csv",
}

# Geographic hold-out: whole flood events are unseen in training. Test spans three continents
# and a range of water fractions; validation uses two further events so early stopping is
# also out-of-event.
GEO_HOLDOUT_V1 = {
    "test": ("Bolivia", "Mekong", "Ghana"),
    "val": ("Pakistan", "Spain"),
}


@dataclass(frozen=True, order=True)
class Chip:
    event: str
    chip_id: str

    @property
    def name(self) -> str:
        return f"{self.event}_{self.chip_id}"

    def s1_path(self, root: Path) -> Path:
        return root / HAND / "S1Hand" / f"{self.name}_S1Hand.tif"

    def label_path(self, root: Path) -> Path:
        return root / HAND / "LabelHand" / f"{self.name}_LabelHand.tif"

    @classmethod
    def from_name(cls, name: str) -> Chip:
        event, sep, chip_id = name.partition("_")
        if not sep or not chip_id or "_" in chip_id:
            raise ValueError(f"unexpected chip name: {name}")
        return cls(event, chip_id)

    @classmethod
    def from_filename(cls, filename: str) -> Chip:
        stem = Path(filename).name
        parts = stem.split("_")
        if len(parts) < 3:
            raise ValueError(f"unexpected Sen1Floods11 filename: {filename}")
        return cls(event=parts[0], chip_id=parts[1])


def discover_chips(root: Path) -> list[Chip]:
    """All chips that have both an S1 image and a hand label."""
    s1 = {Chip.from_filename(p.name) for p in (root / HAND / "S1Hand").glob("*_S1Hand.tif")}
    lab = {Chip.from_filename(p.name) for p in (root / HAND / "LabelHand").glob("*_LabelHand.tif")}
    return sorted(s1 & lab)


def read_split_csv(path: Path) -> list[Chip]:
    with path.open(newline="") as f:
        return sorted(Chip.from_filename(row[0]) for row in csv.reader(f) if row)


def official_splits(root: Path) -> dict[str, list[Chip]]:
    return {k: read_split_csv(root / SPLIT_DIR / v) for k, v in OFFICIAL_SPLITS.items()}


def event_holdout_splits(
    chips: Iterable[Chip], test_events: Sequence[str], val_events: Sequence[str]
) -> dict[str, list[Chip]]:
    test_set, val_set = set(test_events), set(val_events)
    if test_set & val_set:
        raise ValueError(f"events in both test and val: {test_set & val_set}")
    chips = list(chips)
    present = {c.event for c in chips}
    missing = (test_set | val_set) - present
    if missing:
        raise ValueError(f"unknown events: {sorted(missing)}")
    out: dict[str, list[Chip]] = {"train": [], "val": [], "test": []}
    for c in chips:
        key = "test" if c.event in test_set else "val" if c.event in val_set else "train"
        out[key].append(c)
    return {k: sorted(v) for k, v in out.items()}


def leave_one_event_out(chips: Iterable[Chip]) -> list[dict[str, Any]]:
    """One fold per event: that event is test, all others train (Phase 8 generalisation study)."""
    chips = sorted(chips)
    events = sorted({c.event for c in chips})
    return [
        {
            "held_out": ev,
            "train": [c for c in chips if c.event != ev],
            "test": [c for c in chips if c.event == ev],
        }
        for ev in events
    ]


def check_no_leakage(splits: dict[str, list[Chip]], *, by_event: bool) -> list[str]:
    """Return problems: chips in two splits, or (if ``by_event``) events shared by splits."""
    problems = []
    keys = list(splits)
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            dup = set(splits[a]) & set(splits[b])
            if dup:
                problems.append(f"{len(dup)} chips in both {a} and {b}")
            if by_event:
                shared = {c.event for c in splits[a]} & {c.event for c in splits[b]}
                if shared:
                    problems.append(f"events shared by {a} and {b}: {sorted(shared)}")
    return problems


def splits_to_json(splits: dict[str, list[Chip]], meta: dict[str, Any]) -> tuple[str, str]:
    """Serialise deterministically; returns (json_text, sha256 of the chip lists)."""
    lists = {k: [c.name for c in v] for k, v in splits.items()}
    digest = hashlib.sha256(json.dumps(lists, sort_keys=True).encode()).hexdigest()
    payload = {"meta": {**meta, "sha256": digest}, "counts": {k: len(v) for k, v in lists.items()}}
    payload["splits"] = lists
    return json.dumps(payload, indent=1, sort_keys=True), digest


def load_splits_json(path: Path) -> dict[str, list[Chip]]:
    data = json.loads(path.read_text())
    lists = data["splits"]
    digest = hashlib.sha256(json.dumps(lists, sort_keys=True).encode()).hexdigest()
    if digest != data["meta"]["sha256"]:
        raise ValueError(f"split file {path} was modified (hash mismatch)")
    return {k: [Chip.from_name(n) for n in v] for k, v in lists.items()}


# --------------------------------------------------------------------------- reading & normalising


def read_chip(chip: Chip, root: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (image (2,H,W) float32 dB, NaN = nodata; target (H,W) uint8 with IGNORE_INDEX)."""
    with rasterio.open(chip.s1_path(root)) as s:
        img = s.read().astype(np.float32)
    with rasterio.open(chip.label_path(root)) as lab:
        raw = lab.read(1)
    if img.shape[1:] != raw.shape:
        raise ValueError(f"{chip.name}: image {img.shape} and label {raw.shape} differ")
    return img, encode_target(raw, img)


def encode_target(raw_label: np.ndarray, image: np.ndarray) -> np.ndarray:
    invalid = (raw_label < 0) | ~np.isfinite(image).all(axis=0)
    target = np.where(raw_label == 1, 1, 0).astype(np.uint8)
    target[invalid] = IGNORE_INDEX
    return target


@dataclass(frozen=True)
class NormStats:
    mean: tuple[float, ...]
    std: tuple[float, ...]
    clip: tuple[float, float]
    n_pixels: int
    source: str

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=1)

    @classmethod
    def from_json(cls, text: str) -> NormStats:
        d = json.loads(text)
        return cls(
            mean=tuple(d["mean"]),
            std=tuple(d["std"]),
            clip=tuple(d["clip"]),
            n_pixels=d["n_pixels"],
            source=d["source"],
        )


def compute_norm_stats(chips: Sequence[Chip], root: Path, source: str) -> NormStats:
    """Per-band mean/std over valid, clipped pixels. Call with the TRAIN split only.
    Uses float64 running sums (exact, single pass)."""
    n = 0
    s = np.zeros(len(BANDS))
    ss = np.zeros(len(BANDS))
    for chip in chips:
        with rasterio.open(chip.s1_path(root)) as src:
            img = src.read().astype(np.float64)
        valid = np.isfinite(img).all(axis=0)
        vals = np.clip(img[:, valid], *DB_CLIP)
        n += vals.shape[1]
        s += vals.sum(axis=1)
        ss += (vals**2).sum(axis=1)
    if n == 0:
        raise ValueError("no valid pixels")
    mean = s / n
    std = np.sqrt(np.maximum(ss / n - mean**2, 1e-12))
    return NormStats(
        tuple(mean.round(6).tolist()), tuple(std.round(6).tolist()), DB_CLIP, int(n), source
    )


def normalize(img_db: np.ndarray, stats: NormStats) -> np.ndarray:
    """Clip, standardise, and set nodata to 0 (= the training mean). Returns float32."""
    x = np.clip(img_db, *stats.clip)
    mean = np.asarray(stats.mean, np.float32)[:, None, None]
    std = np.asarray(stats.std, np.float32)[:, None, None]
    x = (x - mean) / std
    return np.nan_to_num(x, nan=0.0).astype(np.float32)


def label_summary(chips: Sequence[Chip], root: Path) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    for c in chips:
        _, t = read_chip(c, root)
        counts["water"] += int((t == 1).sum())
        counts["dry"] += int((t == 0).sum())
        counts["ignore"] += int((t == IGNORE_INDEX).sum())
    labelled = counts["water"] + counts["dry"]
    return {
        **counts,
        "water_fraction_of_labelled": counts["water"] / labelled if labelled else None,
        "ignore_fraction": counts["ignore"] / max(1, sum(counts.values())),
    }


# --------------------------------------------------------------------------- torch Dataset


class Sen1Floods11Dataset:
    """PyTorch-compatible map-style dataset. Returns dict(image FloatTensor[2,H,W],
    mask LongTensor[H,W] with IGNORE_INDEX, chip str).

    ``transform`` (e.g. Albumentations) receives ``image`` as (H,W,C) float32 and ``mask``;
    it is applied after normalisation so geometric augmentations see standardised values."""

    def __init__(
        self,
        chips: Sequence[Chip],
        root: Path,
        stats: NormStats,
        transform: Any = None,
    ) -> None:
        self.chips = list(chips)
        self.root = Path(root)
        self.stats = stats
        self.transform = transform

    def __len__(self) -> int:
        return len(self.chips)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        import torch

        chip = self.chips[idx]
        img_db, target = read_chip(chip, self.root)
        img = normalize(img_db, self.stats)
        if self.transform is not None:
            out = self.transform(image=img.transpose(1, 2, 0), mask=target)
            img, target = out["image"].transpose(2, 0, 1), out["mask"]
        return {
            "image": torch.from_numpy(np.ascontiguousarray(img)),
            "mask": torch.from_numpy(np.ascontiguousarray(target).astype(np.int64)),
            "chip": chip.name,
        }
