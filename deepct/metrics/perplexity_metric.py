import torch
import torch.nn.functional as F
from .base import BaseMetric
from .registry import register_metric
from ..tools import logger

@register_metric()
class PerplexityMetric(BaseMetric):
    """
    PerplexityMetric
    ------------------
    计算语言模型在当前输入上的平均困惑度（Perplexity）。

    定义：
        - 困惑度 PPL = exp( CrossEntropyLoss )
        - 每次 forward 时，计算当前 batch 的 token-level CrossEntropyLoss
        - 累积所有 batch 的结果，最后输出平均困惑度
    
    使用：
        dc = DeepCT(model, metrics=["perplexity_metric"])
    """

    name = "perplexity_metric"

    target_layers = "lm_head"
    
    def __init__(self):
        super().__init__()
        self.total_loss = 0.0
        self.total_tokens = 0

    @torch.inference_mode()
    def update(self, layer_name, hidden_states, **kwargs):
        """hidden_states 在最后一层时通常是 logits: [batch, seq, vocab]"""
        logits = hidden_states

        labels = kwargs.get("labels", None)  # 在推理的时候传入 labels
        if labels is None:
            logger.warning(f"[PerplexityMetric] No labels passed from layer {layer_name}, skip.")
            return

        # 对齐输入输出长度（shift trick）
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="mean"
        )

        self.total_loss += loss.item() * shift_labels.numel()
        self.total_tokens += shift_labels.numel()

        logger.debug(f"[PerplexityMetric] layer='{layer_name}' loss={loss.item():.4f}")

    

    def compute(self):
        """返回最终困惑度"""
        if self.total_tokens == 0:
            return {"ppl": float('nan')}
        avg_loss = self.total_loss / self.total_tokens
        ppl = torch.exp(torch.tensor(avg_loss))
        self.values["perplexity"] = ppl.item()
        return self.values