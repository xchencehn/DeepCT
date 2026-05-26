import math

from transformers import AutoModelForCausalLM, AutoTokenizer
from deepct import DeepCT

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto")

dc = DeepCT(model, metrics=["activation_energy_retention"])

prompt = "Please give me a brief introduction to large language models."
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

_ = dc(**inputs)
results = dc.collect()

aer_results = results["activation_energy_retention"]
assert len(aer_results) > 0, "activation_energy_retention produced no entries"

sorted_items = sorted(
    aer_results.items(), key=lambda kv: int(kv[0].split(".")[2])
)
first_layer_name, first_val = sorted_items[0]
assert math.isnan(first_val), (
    f"Expected NaN for first layer AER, got {first_val} at {first_layer_name}"
)

for layer, val in sorted_items[1:]:
    assert val == val, f"NaN AER at {layer}"
    assert val > 0.0, f"Non-positive AER at {layer}: {val}"
    print(f"{layer}: AER={val:.4f}")
