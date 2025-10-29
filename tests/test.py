from transformers import AutoTokenizer, AutoModelForCausalLM
from deepct import DeepCT
import os

# os.environ["LOG_LEVEL"] = "DEBUG"

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto")

dc = DeepCT(model, metrics=["selfattn_repr_correlation", "selfattn_cov_spectrum", "perplexity_metric", "layerwise_perplexity_metric"])

prompt = "Please give me a brief introduction to large language models."
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
_ = dc(**model_inputs)

metrics = dc.collect()
