import torch
from .collector import Collector
from .report import Report
from .metrics import get_metric_instance

class DeepCT(torch.nn.Module):
    def __init__(self, model, metrics, layers="all", verbose=True):
        super().__init__()
        self.model = model
        self.collector = Collector([get_metric_instance(m) for m in metrics])
        self.layers = layers
        self.verbose = verbose
        self._register_hooks()

    def _register_hooks(self):
        # 遍历底层模型层
        try:
            named_layers = self.model.model.layers.named_children()
        except AttributeError:
            named_layers = self.model.named_children()

        for name, module in named_layers:
            module.register_forward_hook(self._hook_fn(name))
            if self.verbose:
                print(f"[DeepCT] Hook registered on {name}")

    def _hook_fn(self, layer_name):
        def hook(module, inputs, outputs):
            hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs
            self.collector.collect(layer_name, hidden_states)
        return hook

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def report(self):
        data = self.collector.compute_all()
        return Report(data)