import torch
import torch.nn.functional as F
from .base import BaseMetric
from .registry import register_metric
from ..tools import logger

@register_metric()
class LayerwisePerplexityMetric(BaseMetric):
    """
    LayerwisePerplexityMetric
    --------------------------
    Computes the "linguistic perplexity" for the output representation h^(l) of each Transformer layer.
    
    Usage:
        dc = DeepCT(model, metrics=["layerwise_perplexity_metric"])

    Principle:
        - For each layer's output h^(l), project it to the vocabulary logits via the model's lm_head;
        - Compute token-level CrossEntropyLoss at each layer;
        - The metric reflects the probabilistic expressiveness of each layer (lower = closer to actual language output).

    Output:
        self.values["model.layers.N"] = ppl value
    """

    name = "layerwise_perplexity_metric"
    target_layers = lambda name: name.count(".") == 2 and name.startswith("model.layers.")

    def __init__(self):
        super().__init__()
        self.total_loss = {}
        self.total_tokens = {}

    @torch.inference_mode()
    def update(self, layer_name, hidden_states, **kwargs):
        labels = kwargs.get("labels", None)
        model = kwargs.get("model", None)
        if labels is None:
            logger.warning(f"[LayerwisePPL] Missing labels for {layer_name}, skip.")
            return
        if model is None:
            logger.warning(f"[LayerwisePPL] Missing model for {layer_name}, skip.")
            return

        if not hasattr(model, "lm_head"):
            logger.warning(f"[LayerwisePPL] Model has no lm_head, skip {layer_name}")
            return
        lm_head = model.lm_head

        logits = lm_head(hidden_states)

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        if shift_labels.numel() == 0:
            return

        # CrossEntropyLoss
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="mean"
        )

        self.total_loss[layer_name] = loss.item()
        self.total_tokens[layer_name] = shift_labels.numel()

    def compute(self):
        results = {}
        for layer, loss in self.total_loss.items():
            ppl = torch.exp(torch.tensor(loss))
            results[layer] = ppl.item()
        self.values = results
        return self.values