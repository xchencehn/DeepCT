import torch

class BaseMetric:
    name = "base"

    target_layers = "all"

    def __init__(self):
        self.values = {}

    def update(self, layer_name, hidden_states, **kwargs):
        # This will be called by the hook of the corresponding layer during each forward pass.
        raise NotImplementedError

    def compute(self):
        # This will be called after all layers have run.
        return self.values


