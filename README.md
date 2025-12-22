## What is DeepCT

<div align="center">
  <img src="img.png"  style="border-radius: 15px;">
</div>



DeepCT enables you to:

- Monitor the dynamics of internal layers in a non-intrusive way;
- Automatically hook outputs from each layer;
- Collect metrics to describe the internal cognitive properties of models;
- Output Summary — letting you know what the DeepCT framework has done;
- Return raw metric results — for you to analyze the internal mechanisms of models in depth.

## Installation and Setup

1 Clone the project and install dependencies:

```
git clone https://github.com/xchencehn/deepct.git
cd deepct
pip install -e .
```

2 Required dependencies:

- Python ≥ 3.8
- PyTorch ≥ 2.0
- Transformers
- Pandas

## Quick Start

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from deepct import DeepCT

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto")

# Step 1 Create a DeepCT instance
dc = DeepCT(model, metrics=["correlator", "intrinsic_dim"])

# Step 2 Prepare input
prompt = "Introduce the principles of large language models."
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

# Step 3 Run Forward (DeepCT will automatically hook)
_ = dc(**inputs)

# Step 4 Generate framework execution summary
dc.summary()         # Print registered metrics, hooked layers, runtime environment, etc.

# Step 5 Collect metric results
metrics = dc.collect()
print(metrics)       # Output a dictionary of layer-wise results for each metric
```

## Summary: Framework Behavior Summary

Calling `dc.summary()` outputs an execution report for this session (not metric results).

Example output:

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

Feature description:

- Runtime Info: timestamp, PyTorch version, model name, number of metrics;
- Registered Metrics: which metrics are loaded;
- Hook Summary: shows which layers each metric hooks into;
- Output is entirely from framework internal behavior (not model content).

## Collecting Metric Results

Call:

```
results = dc.collect()
```

Output structure:

```json
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

> Returns raw result dictionaries for each metric, allowing users to perform plotting, clustering, or analysis on their own.

## Custom Metrics

```python
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

After registration, you can use it directly in DeepCT:

```
dc = DeepCT(model, metrics=["activation_energy"])
```