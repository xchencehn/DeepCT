import torch

from .base import BaseMetric
from .registry import register_metric
from ..tools import logger


@register_metric()
class AttentionHeadEntropy(BaseMetric):
    """
    AttentionHeadEntropy
    ---------------------
    Average Shannon entropy of the per-head attention weight distributions:

        Hₕ      = − Σⱼ aₕ,ⱼ · log aₕ,ⱼ        (per-head, per-query entropy)
        AHE(l)  = (1 / H) · Σₕ Hₕ              (layer-mean across heads)

    High entropy → diffuse / spread-out attention.
    Low  entropy → focused on a few positions.

    Requires the model to expose attention weights. DeepCT auto-sets
    `output_attentions=True` for the forward call when this metric is
    registered, and the framework hook forwards the raw tuple via the
    `raw_outputs` kwarg.

    Usage:
        dc = DeepCT(model, metrics=["attention_head_entropy"])
    """

    name = "attention_head_entropy"
    target_layers = "model.layers.*.self_attn"
    requires_attention = True

    @torch.inference_mode()
    def update(self, layer_name, hidden_states, **kwargs):
        raw_outputs = kwargs.get("raw_outputs", None)
        if not isinstance(raw_outputs, tuple) or len(raw_outputs) < 2:
            logger.warning(
                f"[AttentionHeadEntropy] Layer {layer_name}: no attention weights "
                f"in raw_outputs (got {type(raw_outputs).__name__}), skip."
            )
            self.values[layer_name] = float("nan")
            return

        attn_weights = raw_outputs[1]
        if attn_weights is None:
            logger.warning(
                f"[AttentionHeadEntropy] Layer {layer_name}: attention weights are None. "
                f"Did the model honor output_attentions=True?"
            )
            self.values[layer_name] = float("nan")
            return

        a = attn_weights.detach()
        if a.dtype in (torch.bfloat16, torch.float16):
            a = a.to(torch.float32)

        # Expected shape: [batch, heads, query, key]
        if a.ndim != 4:
            logger.warning(
                f"[AttentionHeadEntropy] Layer {layer_name}: unexpected attention "
                f"shape {tuple(a.shape)}, skip."
            )
            self.values[layer_name] = float("nan")
            return

        eps = 1e-12
        # Entropy along the key axis, then mean over batch×heads×query
        entropy = -(a * torch.log(a + eps)).sum(dim=-1)
        ahe = entropy.mean().item()
        self.values[layer_name] = ahe
