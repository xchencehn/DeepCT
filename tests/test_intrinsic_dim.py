from transformers import AutoModelForCausalLM, AutoTokenizer
from deepct import DeepCT

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto")

dc = DeepCT(model, metrics=["intrinsic_dim"])

prompt = "Please give me a brief introduction to large language models."
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

_ = dc(**inputs)
results = dc.collect()

id_results = results["intrinsic_dim"]
assert len(id_results) > 0, "intrinsic_dim produced no entries"
for layer, val in id_results.items():
    assert val == val, f"NaN intrinsic_dim at {layer}"
    assert val >= 0.0, f"Negative intrinsic_dim at {layer}: {val}"
    print(f"{layer}: ID={val:.4f}")
