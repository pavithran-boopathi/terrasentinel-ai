import pytest
import torch
from torch import nn

from ml.models.factory import ARCHS, adapt_first_conv, build_model, model_from_config

pytest.importorskip("segmentation_models_pytorch")
pytest.importorskip("transformers")


def test_adapt_first_conv_preserves_response_to_constant_input():
    conv = nn.Conv2d(3, 8, 3, padding=1)
    new = adapt_first_conv(conv, 2)
    assert new.weight.shape == (8, 2, 3, 3)
    x3 = torch.ones(1, 3, 5, 5)
    x2 = torch.ones(1, 2, 5, 5)
    # a constant image gives the same activations: sum over channels is preserved
    assert torch.allclose(conv(x3), new(x2), atol=1e-5)


@pytest.mark.parametrize("arch", ARCHS)
def test_all_archs_output_full_resolution_logits(arch):
    kw = {"base_channels": 4, "depth": 2} if arch == "unet" else {}
    m = build_model(arch, pretrained=False, **kw).eval()
    with torch.no_grad():
        out = m(torch.randn(1, 2, 64, 64))
    assert out.shape == (1, 1, 64, 64)


def test_model_from_config_roundtrip():
    cfg = {"arch": "segformer_b0", "dropout": 0.1}
    m1 = model_from_config(cfg)
    m2 = model_from_config(cfg)
    m2.load_state_dict(m1.state_dict())  # same architecture → loads cleanly
    with pytest.raises(ValueError, match="unknown arch"):
        build_model("resnet9000")
