from typing import Callable, ClassVar, Union
LayerSelector = Union[str, Callable[[str], bool], Callable[[str], bool]]

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

    The `update(layer_name, hidden_states, **kwargs)` method is all you need to implement.
    """

    name = "base"

    target_layers: ClassVar[LayerSelector] = "all"

    def __init__(self):
        self.values = {}

    def update(self, layer_name, hidden_states, **kwargs):
        # This will be called by the hook of the corresponding layer during each forward pass.
        raise NotImplementedError

    def compute(self):
        # This will be called after all layers have run.
        return self.values