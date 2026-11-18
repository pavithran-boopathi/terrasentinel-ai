"""Losses for imbalanced binary segmentation with an ignore index."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ml.evaluation.metrics import IGNORE_INDEX


class BCEDiceLoss(nn.Module):
    """``bce_weight * BCE(pos_weight) + dice_weight * (1 - soft Dice)`` over valid pixels only.

    Soft Dice is computed over the whole batch (not averaged per image), which is stable when
    some chips contain no water.
    """

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        pos_weight: float | None = None,
        ignore_index: int = IGNORE_INDEX,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.register_buffer(
            "pos_weight", torch.tensor([pos_weight]) if pos_weight is not None else None
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """logits: (B,1,H,W) or (B,H,W); target: (B,H,W) long with {0,1,ignore}."""
        if logits.dim() == 4:
            logits = logits[:, 0]
        valid = target != self.ignore_index
        if not valid.any():
            return logits.sum() * 0.0  # keeps graph; no signal in this batch
        x = logits[valid].float()
        y = target[valid].float()
        loss = x.new_zeros(())
        if self.bce_weight:
            pw = self.pos_weight.to(x.device) if self.pos_weight is not None else None
            loss = loss + self.bce_weight * F.binary_cross_entropy_with_logits(x, y, pos_weight=pw)
        if self.dice_weight:
            p = torch.sigmoid(x)
            inter = (p * y).sum()
            dice = (2 * inter + self.smooth) / (p.sum() + y.sum() + self.smooth)
            loss = loss + self.dice_weight * (1 - dice)
        return loss
