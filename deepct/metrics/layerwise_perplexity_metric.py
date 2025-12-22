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
    对 Transformer 每一层的输出表示 h^(l)，计算其“语言困惑度”。
    用法：
        dc = DeepCT(model, metrics=["layerwise_perplexity_metric"])

    原理：
        - 对每层输出 h^(l)，通过模型的 lm_head 投影到词表；
        - 在各层计算 token-level CrossEntropyLoss；
        - 指标反映不同层的概率表征能力（越低 = 越接近语言输出）。

    输出：
        self.values["model.layers.N"] = ppl 数值
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

        # 获取词表映射头
        if not hasattr(model, "lm_head"):
            logger.warning(f"[LayerwisePPL] Model has no lm_head, skip {layer_name}")
            return
        lm_head = model.lm_head

        # 得到 logits: [batch, seq, vocab]
        logits = lm_head(hidden_states)

        # shift trick 对齐标签
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