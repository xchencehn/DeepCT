from transformers import AutoModelForCausalLM, AutoTokenizer
from deepct import DeepCT

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    attn_implementation="eager",
)

dc = DeepCT(model, metrics=["attention_head_entropy"])

prompt = "Please give me a brief introduction to large language models."
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

_ = dc(**inputs)
results = dc.collect()

ahe_results = results["attention_head_entropy"]
assert len(ahe_results) > 0, "attention_head_entropy produced no entries"
for layer, val in ahe_results.items():
    assert val == val, f"NaN AHE at {layer}"
    assert val >= 0.0, f"Negative AHE at {layer}: {val}"
    print(f"{layer}: AHE={val:.4f}")
