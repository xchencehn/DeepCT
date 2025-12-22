import torch
from .base import BaseMetric
from .registry import register_metric
from ..tools import logger

@register_metric()
class SelfAttentionCovarianceSpectrum(BaseMetric):
    """
    SelfAttentionCovarianceSpectrum
    -------------------
    Analyze the covariance eigenvalue spectrum (Covariance Spectrum) of the outputs from Transformer self-attention modules.

    Metric Goals:
        - Reveal the energy distribution and representational diversity of outputs from each self-attention layer;
        - Assess whether information is compressed, diffused, or redundant within a layer based on the spectral shape;
        - Provide fundamental data for subsequent intrinsic dimensionality, energy compression ratio, and related metrics.

    Calculation Steps:
        1. Mean-center the output representations h (remove the mean);
        2. Compute the covariance matrix C = (hᵀh)/(N-1);
        3. Compute the eigenvalue spectrum of C;
        4. Retain only eigenvalues greater than 1e-8.

    Output:
        self.values[layer_name] = tensor([...])
        Represents the covariance spectrum (variance energy in each direction) for the given layer.

    Typical Use Cases:
        - Analysis of spatial structure of layer representations;
        - Information compression and degradation detection;
        - Modeling with spectral/energy-based metrics.
    """

    name = "selfattn_cov_spectrum"
    target_layers = "model.layers.*.self_attn"

    def update(self, layer_name, h, **kwargs):
        h = h.detach()
        if h.ndim > 2:
            h = h.reshape(-1, h.size(-1))  # Merge batch and sequence dimensions

        if h.dtype == torch.bfloat16:
            h = h.to(torch.float32)

        h_centered = h - h.mean(dim=0, keepdim=True)
        n_samples = h_centered.size(0)

        if n_samples <= 1:
            logger.warning(f"[SelfAttnCovSpectrum] Layer {layer_name}: insufficient sample size (n_samples={n_samples}), returning empty tensor")
            self.values[layer_name] = torch.tensor([])
            return

        cov = (h_centered.T @ h_centered) / (n_samples - 1)

        try:
            eig = torch.linalg.eigvals(cov).real
            eig = eig[torch.abs(eig) > 1e-8]
        except RuntimeError as e:
            logger.warning(f"[SelfAttnCovSpectrum] Layer {layer_name}: eigenvalue computation failed - {str(e)}, returning empty tensor")
            self.values[layer_name] = torch.tensor([])
            return

        self.values[layer_name] = eig