from deepct import DeepCT
from transformers import AutoTokenizer, AutoModelForCausalLM

DeepCT.list_metrics()


model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto"
)
dc = DeepCT(model, metrics=["intrisic_dimm"])  # 写错
