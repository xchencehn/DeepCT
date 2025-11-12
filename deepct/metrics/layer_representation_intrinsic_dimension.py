# -*- coding: utf-8 -*-
import torch
from .base import BaseMetric
from .registry import register_metric
from ..tools import logger

@register_metric()
class LayerRepresentationIntrinsicDimension(BaseMetric):
    """
    LayerRepresentationIntrinsicDimension
    -------------------------------------
    计算 Transformer **每个 Block 输出**表征的内在维度 (Intrinsic Dimension)，
    采用 Participation Ratio:  PR = (Σ s_i^2)^2 / Σ s_i^4
    其中 s_i 为隐藏表示矩阵的奇异值（s_i >= 0）。
    """

    # 让框架按这个名字注册与显示（不再出现 base）
    name = "intrinsic_dim"

    # 只在每个 Block 的输出处挂钩（避免把 q/k/v/mlp/ln 等全部子层都算进去）
    target_layers = "model.layers.*"   # 适配 LLaMA/Qwen 结构；如是别家，可按需改成 encoder.layer.* 等

    def __init__(
        self,
        max_tokens: int = 4096,   # 每层最多采样多少个 token，避免 OOM
        center: bool = True,      # 是否去中心化
        use_fp16: bool = False,   # 需要时可降低中间精度
    ):
        super().__init__()
        self.max_tokens = int(max_tokens)
        self.center = bool(center)
        self.use_fp16 = bool(use_fp16)

    # --- 工具：把各种输出规整为 [N, D] 张量 ---
    @torch.no_grad()
    def _to_2d(self, h):
        # tuple/list: 取第0个
        if isinstance(h, (tuple, list)):
            h = h[0] if len(h) else None

        # HF ModelOutput
        if hasattr(h, "last_hidden_state"):
            h = h.last_hidden_state
        elif hasattr(h, "hidden_states"):
            hs = h.hidden_states
            if isinstance(hs, (tuple, list)) and len(hs) > 0:
                h = hs[-1]

        if not isinstance(h, torch.Tensor):
            return None

        if h.ndim == 3:               # [B, T, D] -> [N, D]
            b, t, d = h.shape
            h = h.reshape(b * t, d)
        elif h.ndim != 2:             # 既不是 [B,T,D] 也不是 [T,D] 就跳过
            return None
        return h

    @torch.no_grad()
    def update(self, layer_name, hidden_states, **kwargs):
        """
        输入:
            layer_name:   当前层名（例如 "model.layers.12"）
            hidden_states: 该层输出，可能是 Tensor / tuple / HF ModelOutput
        输出:
            self.values[layer_name] = float(ID)
        """
        h = self._to_2d(hidden_states)
        if h is None or h.numel() == 0:
            return

        # 统一精度
        if h.dtype in (torch.bfloat16, torch.float16):
            h = h.to(torch.float32)
        if self.use_fp16:
            h = h.half()

        # 去中心化
        if self.center:
            h = h - h.mean(dim=0, keepdim=True)

        # 限制样本量
        if h.size(0) > self.max_tokens:
            h = h[: self.max_tokens]

        # 计算奇异值；失败降级到 double
        try:
            s = torch.linalg.svdvals(h)
        except RuntimeError as e:
            logger.warning(f"[IntrinsicDim] svdvals fail at {layer_name}: {e}; fallback to float64")
            s = torch.linalg.svdvals(h.to(torch.float64))

        # Participation Ratio
        s2 = (s ** 2).sum()
        s4 = (s ** 4).sum() + 1e-12
        id_val = float((s2 * s2) / s4)

        self.values[layer_name] = id_val  # ✅ 与其它指标一致的写法，Collector 直接拿 values
