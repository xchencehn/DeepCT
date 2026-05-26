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

## Built-in Metrics

Each entry below lists the metric `name` (the string passed to
`DeepCT(model, metrics=[...])`), the layers it hooks, and what it computes.

### Geometric / Representational

These probe the shape and dimensionality of a layer's hidden representations.

#### `intrinsic_dim` — Intrinsic Dimension (ID)

Effective number of independent directions used by a layer, computed as the
participation ratio of the per-layer covariance spectrum:

    ID(l) = (Σᵢ λᵢ)² / Σᵢ λᵢ²

where `λᵢ` are eigenvalues of `Cₗ = (Hᵀ H) / (N − 1)` in descending order and
`d` is the hidden dimension. Higher → richer, more full-rank representation;
lower → representations are squeezed into fewer directions.

- **Target layers:** `model.layers.<N>` (block output)
- **Output:** one scalar per layer

#### `dimension_collapse_rate` — Dimension Collapse Rate (DCR)

Complement of the normalized effective rank of the covariance spectrum:

    pᵢ      = λᵢ / Σⱼ λⱼ
    erank   = exp(− Σᵢ pᵢ log pᵢ)
    DCR(l)  = 1 − erank(Cₗ) / d

Higher DCR (→ 1) ⇒ stronger collapse; lower DCR (→ 0) ⇒ energy spread across
most directions.

- **Target layers:** `model.layers.<N>`
- **Output:** one scalar per layer

#### `selfattn_repr_correlation` — SelfAttention Representational Correlation E(ξ)

Mean inter-token correlation of self-attention outputs:

    E(ξ) = Σ_{i≠j} (xᵢ · xⱼ) / [ N · Σᵢ ‖xᵢ‖² ]

High ⇒ tokens converge to similar representations (potential redundancy);
low ⇒ representations stay diverse.

- **Target layers:** `model.layers.<N>.self_attn`
- **Output:** one scalar per layer

#### `activation_energy_retention` — Activation Energy Retention (AER)

Ratio of L2 energy between adjacent layers — how much representational energy
survives each layer transition:

    AER(l) = ‖hₗ‖₂² / ‖hₗ₋₁‖₂²

- **Target layers:** `model.layers.<N>` (cross-layer; layer 0 is undefined)
- **Output:** one scalar per layer (from layer 1 onward)

### Information Flow / Activation

#### `activation_sparsity` — Activation Sparsity Rate (ASR)

Fraction of activation units whose magnitude exceeds threshold `τ`:

    ASR(l) = (1 / |hₗ|) · Σᵢ 𝟙[ |hₗ,ᵢ| > τ ]

Default `τ = 1e-8`; configurable (absolute, or as a fraction of the layer-mean
activation). Low values ⇒ activations are sparse / concentrated; high values ⇒
activations are broadly engaged.

- **Target layers:** `model.layers.<N>`
- **Output:** one scalar per layer

#### `attention_head_entropy` — Attention Head Entropy (AHE)

Average Shannon entropy of the per-head attention distributions:

    Hₕ      = − Σⱼ aₕ,ⱼ · log aₕ,ⱼ        (per-head entropy)
    AHE(l)  = (1 / H) · Σₕ Hₕ              (layer-mean across H heads)

High entropy ⇒ diffuse, spread-out attention; low entropy ⇒ focused on a few
positions. Requires the model to expose `attention_weights`, so DeepCT
auto-enables `output_attentions=True` on the forward call when this metric is
registered.

- **Target layers:** `model.layers.<N>.self_attn`
- **Output:** one scalar per layer

### Language Modeling

#### `perplexity_metric` — Overall Perplexity

`PPL = exp(CE_loss)` computed at `lm_head`.

- **Target layers:** `lm_head`
- **Output:** single scalar `perplexity`

#### `layerwise_perplexity_metric` — Per-layer Perplexity

Projects each transformer block's hidden state through `lm_head` and reports a
per-layer PPL — a proxy for how language-shaped each layer's representation is.

- **Target layers:** `model.layers.<N>`
- **Output:** one scalar per layer

### Spectral

#### `selfattn_cov_spectrum` — Covariance Eigenvalue Spectrum

Returns the (filtered) eigenvalue spectrum of each self-attention output's
covariance matrix. Useful as a raw signal feeding `intrinsic_dim`,
`dimension_collapse_rate`, and other spectral analyses.

- **Target layers:** `model.layers.<N>.self_attn`
- **Output:** 1-D tensor per layer

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