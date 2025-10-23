import torch

class BaseMetric:
    name = "base"

    target_layers = "all"

    def __init__(self):
        self.values = {}

    def update(self, layer_name, hidden_states, **kwargs):
        # 这里会在每次 forward 时，被对应层的 hook 调用
        raise NotImplementedError

    def compute(self):
        # 这里会在所有层跑完后调用
        return self.values


