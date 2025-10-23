import torch
from .base import BaseMetric, register_metric

@register_metric("layer_corr")
class LayerCorrelation(BaseMetric):
    name = "layer_corr"
    _prev_h = None

    def update(self, layer_name, h, **kwargs):
        h = h.detach().flatten(1)
        if self._prev_h is not None:
            corr = torch.cosine_similarity(self._prev_h, h, dim=-1).mean().item()
            self.values[layer_name] = corr
        self._prev_h = h