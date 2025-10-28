from transformers import AutoModelForCausalLM, AutoTokenizer
from deepct import DeepCT

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

dc = DeepCT(model, metrics=["perplexity_metric"])

text = "Please give me a brief introduction to large language models."
inputs = tokenizer(text, return_tensors="pt").to(model.device)
inputs["labels"] = inputs["input_ids"]

_ = dc(**inputs)
results = dc.collect()
print(results)