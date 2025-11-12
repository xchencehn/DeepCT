# -*- coding: utf-8 -*-
import torch
import numpy as np
from typing import Dict, List, Optional
from .base import BaseMetric
from .registry import register_metric

@torch.no_grad()
def _participation_ratio(svals: torch.Tensor) -> float:
    """
    Participation Ratio (PR):
      PR = (sum(s^2))^2 / sum(s^4)
    这里用奇异值 s（>=0）计算“有效维度”。
    """
    s2 = (svals ** 2).sum()
    s4 = (svals ** 4).sum() + 1e-12
    return float((s2 * s2) / s4)

@register_metric()
class LayerRepresentationIntrinsicDimension(BaseMetric):
    """
    LayerRepresentationIntrinsicDimension
    ====================================
    计算模型各层隐藏表示（Representation）的内在维度（Intrinsic Dimension）。

    一、指标定义
    ------------
    内在维度用于衡量一层输出的“有效维度”，反映该层表征在高维空间中所占用的子空间复杂度。
    当特征值分布较尖时，表征更“低维”；当特征值分布更平坦时，表征更“高维”。

    常用公式：
      ID = (Σ λ_i)^2 / Σ(λ_i^2)
    其中 λ_i 为协方差矩阵的特征值（等价于奇异值平方）。

    二、输入说明
    ------------
    每层通过 `update(layer_name, hidden_states)` 接收输入：
        layer_name : str
            当前层名称，例如 "encoder.layer.12"
        hidden_states : torch.Tensor
            当前层的隐藏表示张量，形状为 [batch, seq_len, hidden_dim] 或 [seq_len, hidden_dim]

    三、输出说明
    ------------
    compute() -> Dict[str, float]
        返回每一层的内在维度值:
        {
            "encoder.layer.0":  56.8,
            "encoder.layer.1":  58.0,
            ...
        }
    """

    # —— 你可以在 __init__ 中增加可调参数 —— #
    def __init__(
        self,
        max_tokens: int = 2048,     # 每层最多采样 token 数（避免 OOM）
        use_fp16: bool = False,     # 是否将隐藏态转为 fp16 存储再计算
        center: bool = True,        # 是否对隐藏态进行去中心化
    ) -> None:
        super().__init__()
        self.max_tokens = int(max_tokens)
        self.use_fp16 = bool(use_fp16)
        self.center = bool(center)

        # layer_name -> List[Tensor]（逐步累积）
        self._buffers: Dict[str, List[torch.Tensor]] = {}

    # BaseMetric 接口：清空状态
    def reset(self) -> None:
        self._buffers.clear()

    # BaseMetric 接口：累积一层一次前向的输出
    @torch.no_grad()
    def update(self, layer_name: str, hidden_states: torch.Tensor) -> None:
        """
        hidden_states: [B, T, D] 或 [T, D]
        """
        if hidden_states is None:
            return

        # 统一到 [N, D]
        if hidden_states.dim() == 3:
            B, T, D = hidden_states.shape
            H = hidden_states.reshape(B * T, D)
        elif hidden_states.dim() == 2:
            H = hidden_states
        else:
            # 形状异常时忽略
            return

        # 去中心化（按特征维度）
        if self.center:
            H = H - H.mean(dim=0, keepdim=True)

        # 采样（限制 token 数，避免 OOM）
        if H.size(0) > self.max_tokens:
            H = H[: self.max_tokens]

        # 降精度以节省显存/内存
        if self.use_fp16:
            H = H.half()

        # 只在 CPU 上累积，防止 GPU 占用
        H = H.detach().cpu()

        if layer_name not in self._buffers:
            self._buffers[layer_name] = []
        self._buffers[layer_name].append(H)

    # BaseMetric 接口：计算最终结果
    @torch.no_grad()
    def compute(self) -> Dict[str, float]:
        """
        返回: { layer_name: intrinsic_dimension_value }
        """
        results: Dict[str, float] = {}

        for ln, chunks in self._buffers.items():
            if not chunks:
                continue

            # 合并本层收集到的所有片段 -> [N, D]
            H = torch.cat(chunks, dim=0)

            # 计算奇异值（更数值稳定）；若失败，退化到 double
            try:
                s = torch.linalg.svdvals(H)
            except RuntimeError:
                s = torch.linalg.svdvals(H.double())

            # 由奇异值计算 Participation Ratio 作为有效维度
            id_val = _participation_ratio(s)
            results[ln] = float(id_val)

        return results

    # BaseMetric 可选接口：方便统一展示/命名
    @property
    def short_name(self) -> str:
        return "intrinsic_dim"

    @property
    def full_name(self) -> str:
        return "layer_representation_intrinsic_dimension"
