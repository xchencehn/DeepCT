import torch
from .collector import Collector
from .metrics import get_metric_instance, list_registered_metrics
from .tools import summary, setup_logger, logger, banner
import weakref, threading
import datetime



class RuntimeContext(threading.local):
    """Thread-safe context storage per forward call"""
    current_context = None

    def __init__(self):
        self.model_ref = None
        self.kwargs = {}
        self.device = "cpu"
        self.hidden_map = {}
        self.labels = None  

    @classmethod
    def start(cls, model, **kwargs):
        ctx = cls()
        ctx.model_ref = weakref.ref(model)
        ctx.kwargs = kwargs
        ctx.device = next(model.parameters()).device
        ctx.labels = kwargs.get("labels", None)
        cls.current_context = ctx
        return ctx

    @classmethod
    def get(cls):
        return getattr(cls, 'current_context', None)

    @classmethod
    def clear(cls):
        cls.current_context = None

class DeepCT(torch.nn.Module):
    def __init__(self, model, metrics):
        super().__init__()
        self.model = model
        self.metrics = [get_metric_instance(m) for m in metrics]
        self.collector = Collector(self.metrics)
        self._hook_handles = []

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        model_name = getattr(getattr(model, "config", None), "_name_or_path", "unknown")
        metric_names = ", ".join([m.name for m in self.metrics])

        banner(model=self.model, metrics=self.metrics)
        logger.info("Initializing DeepCT at {}", timestamp)
        logger.info("Loaded model: {}", model_name)
        logger.info("Registered metrics: {}", metric_names)

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
            extra["model"] = self.model
            extra["labels"] = ctx.labels

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
        if "labels" not in kwargs and "input_ids" in kwargs:
            kwargs["labels"] = kwargs["input_ids"].clone()
            logger.info("Auto-generated labels from input_ids")

        logger.debug("Model forward started.")
        RuntimeContext.start(self.model, **kwargs)
        out = self.model(*args, **kwargs)
        RuntimeContext.clear()
        return out

    def collect(self):
        """Collect all registered metric results."""
        metrics = self.collector.collect()
        self._summary(metrics)
        failed_metrics = [m.name for m in self.metrics if self.collector.status[m.name]["error_flag"]]
        if failed_metrics:
            logger.warning("Some metrics failed: {}", failed_metrics)
            logger.warning("See detailed trace in deepct_runtime.log.")
        return metrics

    def _summary(self, metrics):
        """Print a user-visible summary report."""
        logger.info("Generating DeepCT summary report...")
        summary(metrics)

    @staticmethod
    def list_metrics():
        """
        List all available Metric names with their brief descriptions.
        """
        return list_registered_metrics()