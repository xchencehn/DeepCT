from typing import Callable, ClassVar, Union
LayerSelector = Union[str, Callable[[str], bool], Callable[[str], bool]]


def is_decoder_block(name: str) -> bool:
    """
    Match exactly one transformer block output (e.g. ``model.layers.0``),
    not its inner submodules like ``model.layers.0.self_attn``.
    """
    return name.count(".") == 2 and name.startswith("model.layers.")


class BaseMetric:
    """
    Base class for all DeepCT metrics.

    Each subclass must define:
      - `name`: unique string key
      - `target_layers`: controls where hooks are registered.
                "all"             → hook every submodule
                "model.layers.*"  → wildcard pattern match
                callable(name)    → custom layer filter

        DeepCT inspects this attribute when initializing
        and hooks the corresponding modules.

      - `requires_attention` (optional): if True, DeepCT will force
        `output_attentions=True` on the forward call so the hook can pass
        the attention-weight tensor through `kwargs["raw_outputs"]`.

    The `update(layer_name, hidden_states, **kwargs)` method is all you need to implement.
    """

    name = "base"

    target_layers: ClassVar[LayerSelector] = "all"
    requires_attention: ClassVar[bool] = False

    def __init__(self):
        self.values = {}

    def update(self, layer_name, hidden_states, **kwargs):
        # This will be called by the hook of the corresponding layer during each forward pass.
        raise NotImplementedError

    def compute(self):
        # This will be called after all layers have run.
        return self.values