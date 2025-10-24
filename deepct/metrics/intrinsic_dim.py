import torch
from .base import BaseMetric
from .registry import register_metric
from ..tools import logger

@register_metric()
class IntrinsicDim(BaseMetric):
    name = "intrinsic_dim"
    target_layers = "model.layers.*.self_attn"

    def update(self, layer_name, h, **kwargs):
        h = h.detach()
        if h.ndim > 2:
            h = h.reshape(-1, h.size(-1))
        
        if h.dtype == torch.bfloat16:
            h = h.to(torch.float32)
        
        h_centered = h - h.mean(dim=0, keepdim=True)
        n_samples = h_centered.size(0)
        
        if n_samples <= 1:
            logger.warning(f"[IntrinsicDim] Layer {layer_name}: 样本量不足 (n_samples={n_samples})，返回空张量")
            self.values[layer_name] = torch.tensor([])
            return
        
        cov = (h_centered.T @ h_centered) / (n_samples - 1)
        
        try:
            eig = torch.linalg.eigvals(cov).real
            eig = eig[torch.abs(eig) > 1e-8]
        except RuntimeError as e:
            logger.warning(f"[IntrinsicDim] Layer {layer_name}: 特征值计算失败 - {str(e)}，返回空张量")
            self.values[layer_name] = torch.tensor([])
            return
        
        self.values[layer_name] = eig