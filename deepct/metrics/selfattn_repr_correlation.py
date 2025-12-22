import torch
from .base import BaseMetric
from .registry import register_metric


@register_metric()
class SelfAttnRepresentationalCorrelation(BaseMetric):
    """
    SelfAttnRepresentationalCorrelation
    -----------------------------------
    Analyzes the global representational correlation between output token embeddings of Transformer self-attention modules.

    Mathematical definition:
        E(ξ) = [ Σ_{i≠j} (x_i · x_j) ] / [ N * Σ_i ||x_i||² ]

    Where:
        - x_i denotes the hidden vector of each token;
        - N is the total number of tokens;
        - The numerator sums the dot products between all token pairs (excluding self-pairs);
        - The denominator is the normalization term, which is the sum of each token's squared norm, scaled by N.

    Physical/representational meaning:
        - Describes the overall similarity among output representations;
        - Higher value → different token representations are more similar (potential redundancy of information);
        - Lower value → representations are more independent and diverse;
        - Can be used to assess the degree of information diffusion vs. homogenization in the layer.
    """

    name = "selfattn_repr_correlation"
    target_layers = "model.layers.*.self_attn"

    @torch.inference_mode()
    def update(self, layer_name, h, **kwargs):
        """Compute the global representational correlation E(ξ) for self-attention layer outputs."""
        # Ensure data dimension consistency
        if h.ndim > 2:
            h = h.reshape(-1, h.size(-1))  # [tokens, hidden_dim]

        # Convert to float (some models may output bfloat16)
        if h.dtype == torch.bfloat16:
            h = h.to(torch.float32)

        # Prevent zero-dim or empty input
        if h.numel() == 0:
            self.values[layer_name] = float("nan")
            return

        # Compute dot-product matrix (Gram matrix)
        G = h @ h.T
        diag_sum = torch.sum(torch.diag(G))  # sum_i ||x_i||²
        total_sum = torch.sum(G)              # sum_{ij} x_i·x_j
        n_tokens = h.size(0)

        # Remove self-correlation terms (i ≠ j)
        cross_sum = total_sum - diag_sum

        # Normalize to get average correlation E(ξ)
        correlator = cross_sum / (n_tokens * diag_sum + 1e-10)
        self.values[layer_name] = float(correlator.detach().cpu().item())

    def compute(self):
        """Return the representational correlation indicator for each layer."""
        return self.values