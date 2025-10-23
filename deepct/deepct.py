import torch
import datetime
from .collector import Collector
from .summary import Summary
from .metrics import get_metric_instance

class DeepCT(torch.nn.Module):
    def __init__(self, model, metrics, verbose=True):
        super().__init__()
        self.model = model
        self.metrics = [get_metric_instance(m) for m in metrics]
        self.collector = Collector(self.metrics)
        self.verbose = verbose
        self._hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        for metric in self.metrics:
            for name, module in self.model.named_modules():
                tl = metric.target_layers
                if tl == "all":
                    pass  # 全部 hook
                elif callable(tl):
                    if not tl(name):
                        continue
                elif isinstance(tl, str):
                    if "*" in tl:
                        from fnmatch import fnmatch
                        if not fnmatch(name, tl):
                            continue
                    elif name != tl:
                        continue

                # 注册 hook
                handle = module.register_forward_hook(self._hook_fn(name))
                self._hook_handles.append(handle)

                if self.verbose:
                    print(f"[DeepCT] {metric.name} hook -> {name}")

    def _hook_fn(self, layer_name):
        def hook(module, inputs, outputs):
            hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs
            self.collector.update(layer_name, hidden_states)
        return hook

    def clear_hooks(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def collect(self):
        return self.collector.collect()

    def summary(self):
        runtime_info = {
            "timestamp": str(datetime.datetime.now()),
            "torch_version": torch.__version__,
            "n_metrics": len(self.metrics),
            "model_name": getattr(self.model.config, "_name_or_path", "unknown"),
        }
        hook_log = [
            {"metric": m.name, "layer": name}
            for m in self.metrics
            for name, module in self.model.named_modules()
            if hasattr(module, "_forward_hooks")
        ]

        summary = Summary(metrics=self.metrics, hook_log=hook_log, runtime_info=runtime_info)
        summary.show()
