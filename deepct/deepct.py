import torch
from .collector import Collector
from .metrics import get_metric_instance
from .tools import summary, logger, print_banner
import weakref, threading

class RuntimeContext(threading.local):
    """Thread-safe context storage per forward call"""
    current_context = None

    def __init__(self):
        self.model_ref = None
        self.kwargs = {}
        self.device = "cpu"
        self.hidden_map = {}

    @classmethod
    def start(cls, model, **kwargs):
        ctx = cls()
        ctx.model_ref = weakref.ref(model)
        ctx.kwargs = kwargs
        ctx.device = next(model.parameters()).device
        cls.current_context = ctx
        return ctx

    @classmethod
    def get(cls):
        return getattr(cls, 'current_context', None)

    @classmethod
    def clear(cls):
        cls.current_context = None


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
            ctx = RuntimeContext.get()
            hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs
            extra = ctx.kwargs if ctx else {}

            self.collector.update(layer_name, hidden_states, **extra)
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
        RuntimeContext.start(self.model, **kwargs)
        out = self.model(*args, **kwargs)
        RuntimeContext.clear()
        return out

    def collect(self):
        logger.info("Collecting computed metric results...")
        data = self.collector.collect()
        logger.success("Metrics collected successfully: {}",
                       ", ".join(data.keys()))
        return data

    def summary(self, metrics):
        summary(metrics)