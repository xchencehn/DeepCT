from transformers import AutoModelForCausalLM, AutoTokenizer
from deepct import DeepCT

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

dc = DeepCT(model, metrics=["layerwise_perplexity_metric"])

text = "Please give me a brief introduction to large language models."
inputs = tokenizer(text, return_tensors="pt").to(model.device)

_ = dc(**inputs)
results = dc.collect()
dc.summary(results)

# 打印困惑度曲线
for k, v in results["layerwise_perplexity_metric"].items():
    print(k, v)