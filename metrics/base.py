import torch

class BaseMetric:
    name = "base"

    def __init__(self):
        self.values = {}

    def update(self, layer_name, hidden_states, **kwargs):
        raise NotImplementedError

    def compute(self):
        return self.values


