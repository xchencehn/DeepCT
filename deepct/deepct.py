import torch
from .logger import logger, print_banner
from .collector import Collector
from .metrics import get_metric_instance
from .tools import summary



class DeepCT(torch.nn.Module):
    def __init__(self, model, metrics, verbose=True):
        super().__init__()
        self.model = model
        self.metrics = [get_metric_instance(m) for m in metrics]
        self.collector = Collector(self.metrics)
        self.verbose = verbose
        self._hook_handles = []

        print_banner(model=self.model, metrics=self.metrics)

        self._register_hooks()
        logger.info("Hook registration completed, total {} hooks", len(self._hook_handles))

    def _register_hooks(self):
        for metric in self.metrics:
            for name, module in self.model.named_modules():
                tl = metric.target_layers
                if tl == "all":
                    pass
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

                handle = module.register_forward_hook(self._hook_fn(name))
                self._hook_handles.append(handle)

                logger.debug("[hook registered] metric='{}' -> layer='{}'", metric.name, name)

    def _hook_fn(self, layer_name):
        def hook(module, inputs, outputs):
            hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs
            self.collector.update(layer_name, hidden_states)
            logger.trace("[hook trigger] layer='{}' updated metrics", layer_name)
        return hook

    def clear_hooks(self):
        for h in self._hook_handles:
            h.remove()
        count = len(self._hook_handles)
        self._hook_handles.clear()
        logger.info("Cleared {} hooks.", count)

    def forward(self, *args, **kwargs):
        logger.debug("Model forward started.")
        return self.model(*args, **kwargs)

    def collect(self):
        logger.info("Collecting computed metric results...")
        data = self.collector.collect()
        logger.success("Metrics collected successfully: {}",
                       ", ".join(data.keys()))
        return data

    def summary(self, metrics):
        summary(metrics)