import torch

from .base import BaseMetric, is_decoder_block
from .registry import register_metric
from ..tools import logger


@register_metric()
class DimensionCollapseRate(BaseMetric):
    """
    DimensionCollapseRate
    ----------------------
    Complement of the normalized effective rank of a transformer block's hidden
    representation covariance:

        pᵢ      = λᵢ / Σⱼ λⱼ
        erank   = exp(− Σᵢ pᵢ log pᵢ)
        DCR(l)  = 1 − erank(Cₗ) / d

    DCR → 1 indicates strong collapse onto a low-dimensional subspace.
    DCR → 0 indicates representation energy is spread across most directions.

    Usage:
        dc = DeepCT(model, metrics=["dimension_collapse_rate"])
    """

    name = "dimension_collapse_rate"
    target_layers = is_decoder_block

    @torch.inference_mode()
    def update(self, layer_name, hidden_states, **kwargs):
        h = hidden_states.detach()
        if h.ndim > 2:
            h = h.reshape(-1, h.size(-1))

        if h.dtype in (torch.bfloat16, torch.float16):
            h = h.to(torch.float32)

        n_samples, hidden_dim = h.size(0), h.size(-1)
        if n_samples <= 1:
            logger.warning(
                f"[DimensionCollapseRate] Layer {layer_name}: insufficient samples "
                f"(n_samples={n_samples}), skip."
            )
            self.values[layer_name] = float("nan")
            return

        h_centered = h - h.mean(dim=0, keepdim=True)

        try:
            singular = torch.linalg.svdvals(h_centered)
        except RuntimeError as e:
            logger.warning(
                f"[DimensionCollapseRate] Layer {layer_name}: svdvals failed - {e}, skip."
            )
            self.values[layer_name] = float("nan")
            return

        eig = (singular ** 2) / (n_samples - 1)
        eig = eig[eig > 1e-12]
        if eig.numel() == 0:
            self.values[layer_name] = 1.0
            return

        p = eig / (eig.sum() + 1e-12)
        entropy = -(p * torch.log(p + 1e-12)).sum()
        erank = torch.exp(entropy).item()
        dcr = 1.0 - erank / hidden_dim
        self.values[layer_name] = float(max(0.0, min(1.0, dcr)))
