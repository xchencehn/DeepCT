import torch

from .base import BaseMetric, is_decoder_block
from .registry import register_metric
from ..tools import logger


@register_metric()
class IntrinsicDim(BaseMetric):
    """
    IntrinsicDim
    -------------
    Effective number of independent directions used by a transformer block's
    hidden representation, computed as the participation ratio of the eigenvalue
    spectrum of the per-layer feature covariance matrix:

        ID(l) = (Σᵢ λᵢ)² / Σᵢ λᵢ²

    Higher values indicate a richer, more full-rank representation; lower values
    indicate that information is concentrated in fewer directions.

    Usage:
        dc = DeepCT(model, metrics=["intrinsic_dim"])
    """

    name = "intrinsic_dim"
    target_layers = is_decoder_block

    @torch.inference_mode()
    def update(self, layer_name, hidden_states, **kwargs):
        h = hidden_states.detach()
        if h.ndim > 2:
            h = h.reshape(-1, h.size(-1))

        if h.dtype in (torch.bfloat16, torch.float16):
            h = h.to(torch.float32)

        n_samples = h.size(0)
        if n_samples <= 1:
            logger.warning(
                f"[IntrinsicDim] Layer {layer_name}: insufficient sample size "
                f"(n_samples={n_samples}), skip."
            )
            self.values[layer_name] = float("nan")
            return

        h_centered = h - h.mean(dim=0, keepdim=True)

        try:
            singular = torch.linalg.svdvals(h_centered)
        except RuntimeError as e:
            logger.warning(
                f"[IntrinsicDim] Layer {layer_name}: svdvals failed - {e}, skip."
            )
            self.values[layer_name] = float("nan")
            return

        eig = (singular ** 2) / (n_samples - 1)
        eig = eig[eig > 1e-12]
        if eig.numel() == 0:
            self.values[layer_name] = 0.0
            return

        numerator = eig.sum().pow(2)
        denominator = (eig ** 2).sum()
        pr = (numerator / (denominator + 1e-12)).item()
        self.values[layer_name] = pr
