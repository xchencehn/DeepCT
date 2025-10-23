## DeepCT：为大模型打造的内部信号探测系统

DeepCT（Deep Computed Tomography）是一套用于大型语言模型（LLM）和 Transformer 架构的内部信号探测与可解释性分析框架。


它像一台为神经网络设计的 CT 设备 —— 非侵入式、可分层地“扫描”模型内部结构与信息流动。


通过 DeepCT，你可以：
- 在不修改模型结构、不重新训练的情况下，直接采集 隐藏状态、注意力权重、残差信号 等内部动态；
- 从信息几何、流形、维度压缩等角度分析模型的 语义分层特征与表示演化；
- 自动生成包括 内在维度、注意力熵、层相关性、能量保真度 等在内的诊断指标报告；
- 以“模型健康体检”的方式，洞察训练质量、泛化性能、与语义表示合理性。
DeepCT 致力于让大模型内部的复杂机制 可视化、可量化、可诊断，
帮助研究者和开发者用科学仪器般的严谨视角理解智能系统的“神经构造”。
DeepCT — Deep inside your model.



### 当前计划的指标：
1. 内在维度 ID 
2. Layer Correlator E(l) 
3. Attention Head 熵
4. 激活能量保持率 
5. Token 相似度分布
6. L2 范数分布

### 快速测试

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from wrapper import deepct

model_name = "Qwen/Qwen2.5-VL-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# deepct_model = deepct(model, metrics=["intrinsic_dim", "attn_entropy"])
deepct_model = deepct(model, metrics=["correlator", ])
# prepare the model input
prompt = "给我简单介绍一下大型语言模型。"
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
outputs = deepct_model(**model_inputs)

report = deepct_model.report()
report.show()
# report.save("./deepct_metrics")
```