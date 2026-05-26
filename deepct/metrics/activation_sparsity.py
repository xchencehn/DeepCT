import torch

from .base import BaseMetric, is_decoder_block
from .registry import register_metric


@register_metric()
class ActivationSparsity(BaseMetric):
    """
    ActivationSparsity
    -------------------
    Fraction of activation units whose magnitude exceeds a threshold τ:

        ASR(l) = (1 / |hₗ|) · Σᵢ 𝟙[ |hₗ,ᵢ| > τ ]

    Low values ⇒ activations are sparse / concentrated (most units near zero).
    High values ⇒ activations are broadly engaged.

    Args:
        threshold: absolute magnitude cutoff. Default 1e-8.

    Usage:
        dc = DeepCT(model, metrics=["activation_sparsity"])
    """

    name = "activation_sparsity"
    target_layers = is_decoder_block

    def __init__(self, threshold=1e-8):
        super().__init__()
        self.threshold = threshold

    @torch.inference_mode()
    def update(self, layer_name, hidden_states, **kwargs):
        h = hidden_states.detach()
        if h.dtype in (torch.bfloat16, torch.float16):
            h = h.to(torch.float32)

        active = (h.abs() > self.threshold).float().mean().item()
        self.values[layer_name] = active
