"""U-Net (Ronneberger et al., 2015) for 2-band SAR input, trained from scratch.

Kept self-contained (no pretrained encoder) so the baseline has no external weight downloads and
the architecture is fully visible. ``dropout`` in the decoder enables MC-dropout uncertainty
estimation later (Phase 8).
"""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Sequential):
    def __init__(self, cin: int, cout: int, dropout: float = 0.0) -> None:
        layers: list[nn.Module] = [
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        super().__init__(*layers)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        num_classes: int = 1,
        base_channels: int = 32,
        depth: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        chs = [base_channels * 2**i for i in range(depth + 1)]
        self.encoders = nn.ModuleList()
        cin = in_channels
        for c in chs[:-1]:
            self.encoders.append(ConvBlock(cin, c))
            cin = c
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(chs[-2], chs[-1], dropout)
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for c_hi, c_lo in zip(reversed(chs[1:]), reversed(chs[:-1]), strict=True):
            self.ups.append(nn.ConvTranspose2d(c_hi, c_lo, 2, stride=2))
            self.decoders.append(ConvBlock(2 * c_lo, c_lo, dropout))
        self.head = nn.Conv2d(chs[0], num_classes, 1)
        self.depth = depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = 2**self.depth
        if x.shape[-1] % m or x.shape[-2] % m:
            raise ValueError(f"input H, W must be divisible by {m}, got {tuple(x.shape[-2:])}")
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips), strict=True):
            x = dec(torch.cat([up(x), skip], dim=1))
        return self.head(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
