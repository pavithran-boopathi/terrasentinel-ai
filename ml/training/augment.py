"""Augmentations for SAR chips (Albumentations 2.0). Only label-preserving, physically plausible
transforms: dihedral flips/rotations, small crops, and mild multiplicative-noise-like jitter in
normalised space. No colour/brightness transforms designed for optical imagery."""

from __future__ import annotations

import albumentations as A


def train_transform(crop: int | None = None, noise_std: float = 0.05) -> A.Compose:
    ops: list = [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5)]
    if crop:
        ops.append(A.RandomCrop(crop, crop))
    if noise_std > 0:
        ops.append(A.GaussNoise(std_range=(0.0, noise_std), mean_range=(0.0, 0.0), p=0.5))
    return A.Compose(ops)
