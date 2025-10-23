import torch
from .base import BaseMetric, register_metric

@register_metric("intrinsic_dim")
class IntrinsicDim(BaseMetric):
    name = "intrinsic_dim"

    def update(self, layer_name, h, **kwargs):
        h = h.detach()
        cov = (h.T @ h) / h.shape[0]
        eig = torch.linalg.eigvals(cov).real
        eig = eig[eig > 1e-8]
        id_val = (eig.sum() ** 2 / (eig ** 2).sum()).item()
        self.values[layer_name] = float(id_val)