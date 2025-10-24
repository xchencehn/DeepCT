import torch
from .base import BaseMetric
from .registry import register_metric
from ..tools import logger

@register_metric()
class SelfAttentionCovarianceSpectrum(BaseMetric):
    """
    SelfAttentionCovarianceSpectrum
    -------------------
    分析 Transformer 自注意力模块输出的 协方差特征值谱分布（Covariance Spectrum）。

    指标目标：
        - 揭示各 self-attention 层输出表征的能量分布与表示多样性；
        - 从谱形态评估信息是否在层内被压缩、扩散或冗余；
        - 可作为后续内在维度、能量压缩率等指标的基础数据。

    计算步骤：
        1. 对层输出表示 h 去均值（中心化）；
        2. 计算协方差矩阵 C = (hᵀh)/(N-1)；
        3. 对 C 求特征值谱；
        4. 保留特征值 > 1e-8 的部分。

    输出：
        self.values[layer_name] = tensor([...])  
        表示该层的协方差谱（各方向的方差能量）。

    典型用途：
        - 层表征空间结构分析；
        - 信息压缩与退化检测；
        - 光谱型指标（spectral）建模。
    """

    name = "selfattn_cov_spectrum"
    target_layers = "model.layers.*.self_attn"

    def update(self, layer_name, h, **kwargs):
        h = h.detach()
        if h.ndim > 2:
            h = h.reshape(-1, h.size(-1))  # 合并批次与序列维度

        if h.dtype == torch.bfloat16:
            h = h.to(torch.float32)

        h_centered = h - h.mean(dim=0, keepdim=True)
        n_samples = h_centered.size(0)

        if n_samples <= 1:
            logger.warning(f"[SelfAttnCovSpectrum] Layer {layer_name}: 样本量不足 (n_samples={n_samples})，返回空张量")
            self.values[layer_name] = torch.tensor([])
            return

        cov = (h_centered.T @ h_centered) / (n_samples - 1)

        try:
            eig = torch.linalg.eigvals(cov).real
            eig = eig[torch.abs(eig) > 1e-8]
        except RuntimeError as e:
            logger.warning(f"[SelfAttnCovSpectrum] Layer {layer_name}: 特征值计算失败 - {str(e)}，返回空张量")
            self.values[layer_name] = torch.tensor([])
            return

        self.values[layer_name] = eig