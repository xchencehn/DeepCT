import torch

class BaseMetric:
    name = "base"

    def __init__(self):
        self.values = {}

    def update(self, layer_name, hidden_states, **kwargs):
        raise NotImplementedError

    def compute(self):
        return self.values


# 注册系统
_METRIC_REGISTRY = {}

def register_metric(name):
    def _reg(cls):
        _METRIC_REGISTRY[name] = cls
        return cls
    return _reg

def get_metric_instance(name):
    if name not in _METRIC_REGISTRY:
        raise ValueError(f"Unknown metric '{name}'. Registered metrics: {list(_METRIC_REGISTRY.keys())}")
    return _METRIC_REGISTRY[name]()