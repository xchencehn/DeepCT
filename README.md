## 一、DeepCT 是什么

DeepCT 让你：

- 非侵入式监测模型内部层的动态；
- 自动挂接（hook） 各层输出；
- 收集指标（metric） 来描述模型的内部认知特性；
- 输出 Summary —— 让你知道 DeepCT 框架做了哪些事；
- 返回原始指标结果 —— 供你深入分析模型内部机制。



## 二、安装与准备

1 克隆项目并安装依赖：

```
git clone https://github.com/xchencehn/deepct.git
cd deepct
pip install -e .
```

2 必要依赖：

- Python ≥ 3.8
- PyTorch ≥ 2.0
- Transformers
- Pandas



## 三、快速上手

```
from transformers import AutoTokenizer, AutoModelForCausalLM
from deepct import DeepCT

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto")

# Step 1 创建 DeepCT 实例
dc = DeepCT(model, metrics=["correlator", "intrinsic_dim"])

# Step 2 准备输入
prompt = "介绍一下大型语言模型的原理。"
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

# Step 3 运行 Forward（DeepCT 会自动 hook）
_ = dc(**inputs)

# Step 4 生成框架运行摘要
dc.summary()         # 打印已注册指标、hook层、运行环境等信息

# Step 5 采集指标结果
metrics = dc.collect()
print(metrics)       # 输出每个指标的层级结果字典
```



## 四、Summary：框架行为摘要

调用 `dc.summary()` 会输出本次会话的执行报告（而非指标结果）。

输出示例：

```
=== DeepCT Framework Report ===

[Runtime Info]
  timestamp: 2025-10-23 21:35:17
  torch_version: 2.4.1
  n_metrics: 2
  model_name: Qwen/Qwen2.5-0.5B-Instruct

[Registered Metrics]
  - intrinsic_dim     target_layers=all
  - correlator        target_layers=model.layers.*

[Hook Summary]
  Total hooks: 28
   1. intrinsic_dim   -> model.layers.0
   2. intrinsic_dim   -> model.layers.1
   ...
  27. correlator      -> model.layers.25
  28. correlator      -> model.layers.26

===============================
```

功能说明：

- Runtime Info：时间戳、PyTorch 版本、模型名称、指标数量；
- Registered Metrics：哪些指标被加载；
- Hook Summary：显示每个指标 hook 到的层；
- 输出完全来自框架内部行为（非模型内容）。



## 五、采集指标结果

调用：

```
results = dc.collect()
```

输出结构：

```
{
  "intrinsic_dim": {
      "model.layers.0": 126.4,
      "model.layers.1": 119.3,
      ...
  },
  "correlator": {
      "model.layers.0": 0.0031,
      "model.layers.1": 0.0028,
      ...
  }
}
```

> 返回的是每个指标的原始结果字典， 用户可以自行做绘图、聚类或分析。



## 六、自定义指标（Metric）

```
from deepct.metrics.base import BaseMetric
from deepct.metrics.registry import register_metric
import torch

@register_metric()
class ActivationEnergy(BaseMetric):
    name = "activation_energy"
    target_layers = "model.layers.*"

    def update(self, layer_name, h, **kwargs):
        energy = torch.mean(h ** 2).item()
        self.values[layer_name] = energy
```

注册后，就可以直接在 DeepCT 中使用：

```
dc = DeepCT(model, metrics=["activation_energy"])
```



## 七、结合分析：如何处理结果

例如，想用 Pandas 快速对比分布：

```
import pandas as pd

results = dc.collect()
df = pd.DataFrame(results)
print(df.head())
```

或自定义绘图：

```
import matplotlib.pyplot as plt

df.plot()
plt.title("Intrinsic Dimension Across Layers")
plt.xlabel("Layer")
plt.ylabel("Metric Value")
plt.show()
```