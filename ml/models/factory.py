"""Build segmentation models by name. All return (B, 1, H, W) logits at input resolution.

* ``unet``           — our U-Net, random init (Phase 3 baseline)
* ``unet_resnet34``  — U-Net with ImageNet ResNet-34 encoder (segmentation_models_pytorch)
* ``segformer_b0`` / ``segformer_b2`` — SegFormer (Xie et al., 2021), ImageNet MiT encoder
  (Hugging Face ``nvidia/mit-b*``) + all-MLP decoder

Pretrained encoders expect 3-channel RGB. For 2-band SAR we replace the first convolution and
initialise it from the mean of the RGB filters (scaled by 3/2 to keep activation magnitude),
the standard way to reuse ImageNet features for non-RGB input.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ml.models.unet import UNet

ARCHS = ("unet", "unet_resnet34", "segformer_b0", "segformer_b2")
SEGFORMER_HF = {"segformer_b0": "nvidia/mit-b0", "segformer_b2": "nvidia/mit-b2"}
# Architecture hyper-parameters (from the SegFormer paper / HF configs) so random-init models can
# be built offline, e.g. in tests or when rebuilding a checkpoint without network access.
SEGFORMER_ARCH = {
    "segformer_b0": {
        "hidden_sizes": [32, 64, 160, 256],
        "depths": [2, 2, 2, 2],
        "decoder_hidden_size": 256,
    },
    "segformer_b2": {
        "hidden_sizes": [64, 128, 320, 512],
        "depths": [3, 4, 6, 3],
        "decoder_hidden_size": 768,
    },
}


def adapt_first_conv(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    new = nn.Conv2d(
        in_channels,
        conv.out_channels,
        conv.kernel_size,
        conv.stride,
        conv.padding,
        conv.dilation,
        conv.groups,
        conv.bias is not None,
    )
    with torch.no_grad():
        w = conv.weight.mean(dim=1, keepdim=True) * (conv.in_channels / in_channels)
        new.weight.copy_(w.repeat(1, in_channels, 1, 1))
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    return new


def replace_input_conv(model: nn.Module, in_channels: int) -> str:
    """Swap the first 3-channel Conv2d (the RGB stem) for an ``in_channels`` version.
    Located by search, so it survives library refactors of module names. Returns its name."""
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d) and mod.in_channels == 3:
            parent_name, _, attr = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, attr, adapt_first_conv(mod, in_channels))
            return name
    raise ValueError("no 3-channel input convolution found")


class SegFormerBinary(nn.Module):
    def __init__(
        self, variant: str, in_channels: int = 2, pretrained: bool = True, dropout: float = 0.1
    ) -> None:
        super().__init__()
        from transformers import SegformerConfig, SegformerForSemanticSegmentation

        name = SEGFORMER_HF[variant]
        if pretrained:
            model = SegformerForSemanticSegmentation.from_pretrained(
                name, num_labels=1, classifier_dropout_prob=dropout
            )
        else:
            cfg = SegformerConfig(
                num_labels=1, classifier_dropout_prob=dropout, **SEGFORMER_ARCH[variant]
            )
            model = SegformerForSemanticSegmentation(cfg)
        replace_input_conv(model, in_channels)
        model.config.num_channels = in_channels
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(pixel_values=x).logits  # (B, 1, H/4, W/4)
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


def build_model(
    arch: str, *, pretrained: bool = True, in_channels: int = 2, **unet_kwargs: Any
) -> nn.Module:
    if arch == "unet":
        return UNet(in_channels, 1, **unet_kwargs)
    if arch == "unet_resnet34":
        import segmentation_models_pytorch as smp

        return smp.Unet(
            "resnet34",
            encoder_weights="imagenet" if pretrained else None,
            in_channels=in_channels,
            classes=1,
        )
    if arch in SEGFORMER_HF:
        return SegFormerBinary(arch, in_channels, pretrained, unet_kwargs.get("dropout", 0.1))
    raise ValueError(f"unknown arch {arch!r}; choose from {ARCHS}")


def model_from_config(cfg: dict[str, Any], *, pretrained: bool | None = None) -> nn.Module:
    """Rebuild the architecture recorded in a checkpoint config (weights loaded separately)."""
    arch = cfg.get("arch", "unet")
    kw = (
        {"base_channels": cfg["base_channels"], "depth": cfg["depth"], "dropout": cfg["dropout"]}
        if arch == "unet"
        else {"dropout": cfg.get("dropout", 0.1)}
    )
    return build_model(arch, pretrained=False if pretrained is None else pretrained, **kw)
