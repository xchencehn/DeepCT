import torch
from metrics.base import BaseMetric
from metrics.registry import register_metric


@register_metric()
class CorrelatorMetric(BaseMetric):
    name = "correlator"

    def update(self, layer_name, h, **kwargs):
        token_embeddings = h
        flat_embeddings = token_embeddings.reshape(-1, token_embeddings.size(-1))
        batch_seq_len = flat_embeddings.size(0)

        # 计算所有两两向量的点积之和（i ≠ j）
        dot_product_sum = torch.sum(flat_embeddings @ flat_embeddings.T) - torch.sum(
            torch.sum(flat_embeddings ** 2, dim=1))

        # 计算所有向量的范数平方和
        norm_square_sum = torch.sum(torch.sum(flat_embeddings ** 2, dim=1))

        # 计算相关器 E(ξ)
        correlator = dot_product_sum / (batch_seq_len * norm_square_sum)
        self.values[layer_name] = float(correlator)