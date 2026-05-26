from transformers import AutoModelForCausalLM, AutoTokenizer
from deepct import DeepCT

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto")

dc = DeepCT(model, metrics=["dimension_collapse_rate"])

prompt = "Please give me a brief introduction to large language models."
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

_ = dc(**inputs)
results = dc.collect()

dcr_results = results["dimension_collapse_rate"]
assert len(dcr_results) > 0, "dimension_collapse_rate produced no entries"
for layer, val in dcr_results.items():
    assert val == val, f"NaN DCR at {layer}"
    assert 0.0 <= val <= 1.0, f"DCR out of [0,1] at {layer}: {val}"
    print(f"{layer}: DCR={val:.4f}")
