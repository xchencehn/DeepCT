from transformers import AutoTokenizer, AutoModelForCausalLM
from deepct import DeepCT
from deepct.metrics import register_metric
from deepct.metrics.base import BaseMetric

@register_metric()
class BadMetric(BaseMetric):
    name = "bad_metric"

    target_layers = "model.layers.*.self_attn"

    def update(self, layer_name, hidden_states, **kwargs):
        raise Exception("bad metric")

    def compute(self):
        return self.values

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto"
)

dc = DeepCT(model, metrics=["selfattn_repr_correlation", "bad_metric", "selfattn_cov_spectrum"])

prompt = "Please give me a brief introduction to large language models."
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
)

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
_ = dc(**model_inputs)

metrics = dc.collect()

dc.summary(metrics)
