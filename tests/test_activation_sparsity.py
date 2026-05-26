from transformers import AutoModelForCausalLM, AutoTokenizer
from deepct import DeepCT

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto")

dc = DeepCT(model, metrics=["activation_sparsity"])

prompt = "Please give me a brief introduction to large language models."
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

_ = dc(**inputs)
results = dc.collect()

asr_results = results["activation_sparsity"]
assert len(asr_results) > 0, "activation_sparsity produced no entries"
for layer, val in asr_results.items():
    assert val == val, f"NaN ASR at {layer}"
    assert 0.0 <= val <= 1.0, f"ASR out of [0,1] at {layer}: {val}"
    print(f"{layer}: ASR={val:.4f}")
