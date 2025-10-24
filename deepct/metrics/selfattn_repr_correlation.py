import torch
from .base import BaseMetric
from .registry import register_metric


@register_metric()
class SelfAttnRepresentationalCorrelation(BaseMetric):
    """
    SelfAttnRepresentationalCorrelation
    -----------------------------------
    分析 Transformer 自注意力模块输出 token 表示之间的 整体相关性（global representational correlation）。

    数学定义：
        E(ξ) = [ Σ_{i≠j} (x_i · x_j) ] / [ N * Σ_i ||x_i||² ]

    其中：
        - x_i 表示每个 token 的隐藏向量；
        - N 为总 token 数量；
        - 分子为所有样本间点积之和；
        - 分母为自身能量（norm²）的归一化项。

    物理/表征含义：
        - 描述输出表示的整体相似性；
        - 值越大 → 不同 token 表示越相似（潜在信息冗余）；
        - 值越小 → 表示更独立、多样性更好；
        - 可用于衡量层信息扩散性 vs. 聚合同质化程度。
    """

    name = "selfattn_repr_correlation"
    target_layers = "model.layers.*.self_attn"

    @torch.inference_mode()
    def update(self, layer_name, h, **kwargs):
        """计算自注意力层输出之间的全局相关度 E(ξ)"""
        # 确保数据维度一致性
        if h.ndim > 2:
            h = h.reshape(-1, h.size(-1))  # [tokens, hidden_dim]

        # 转浮点类型（部分模型可能输出 bfloat16）
        if h.dtype == torch.bfloat16:
            h = h.to(torch.float32)

        # 防止零维或空输入
        if h.numel() == 0:
            self.values[layer_name] = float("nan")
            return

        # 计算点积矩阵（Gram matrix）
        G = h @ h.T
        diag_sum = torch.sum(torch.diag(G))  # ∑ ||x_i||²
        total_sum = torch.sum(G)              # ∑∑ x_i·x_j
        n_tokens = h.size(0)

        # 去掉自相关项（i ≠ j）
        cross_sum = total_sum - diag_sum

        # 归一化求平均相关度 E(ξ)
        correlator = cross_sum / (n_tokens * diag_sum + 1e-10)
        self.values[layer_name] = float(correlator.detach().cpu().item())

    def compute(self):
        """返回各层的相关指标"""
        return self.values