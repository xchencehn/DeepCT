import datetime
import threading
import weakref
import inspect
from collections import defaultdict

import torch

from .collector import Collector
from .metrics import get_metric_instance, list_registered_metrics
from .tools import banner, logger, setup_logger, summary


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
        return getattr(cls, "current_context", None)

    @classmethod
    def clear(cls):
        cls.current_context = None


def _layer_matches(pattern, name):
    if pattern == "all":
        return True
    if callable(pattern):
        if inspect.ismethod(pattern):
            func = pattern.__func__
            try:
                sig = inspect.signature(func)
            except (TypeError, ValueError):
                pass
            else:
                positional_params = [
                    p
                    for p in sig.parameters.values()
                    if p.kind in (
                        p.POSITIONAL_ONLY,
                        p.POSITIONAL_OR_KEYWORD,
                    )
                ]
                if len(positional_params) == 1:
                    return bool(func(name))
        return bool(pattern(name))
    if isinstance(pattern, str):
        if "*" in pattern:
            from fnmatch import fnmatch
            return fnmatch(name, pattern)
        return name == pattern
    return False


class DeepCT(torch.nn.Module):
    def __init__(
        self,
        model,
        metrics,
    ):
        super().__init__()
        setup_logger()
        self.model = model
        self.metrics = [get_metric_instance(m) for m in metrics]
        self.collector = Collector(self.metrics)

        self._hook_handles = []
        self._hook_plan = {}

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        model_name = getattr(getattr(model, "config", None), "_name_or_path", "unknown")
        metric_names = ", ".join([m.name for m in self.metrics]) or "none"

        banner(model=self.model, metrics=self.metrics)
        logger.info("Initializing DeepCT at {}", timestamp)
        logger.info("Loaded model: {}", model_name)
        logger.info("Registered metrics: {}", metric_names)

        if self.metrics:
            self._register_hooks()
        else:
            logger.warning("No metrics registered. Hooks will not be attached.")

    def _build_hook_plan(self):
        named_modules = dict(self.model.named_modules())
        plan = defaultdict(list)

        for metric in self.metrics:
            for name, module in named_modules.items():
                if _layer_matches(metric.target_layers, name):
                    plan[name].append(metric)

        self._hook_plan = {
            name: {"module": self.model.get_submodule(name), "metrics": metrics}
            for name, metrics in plan.items()
        }

    def _register_hooks(self):
        if not self.metrics:
            logger.warning("No metrics registered. Skip hooking.")
            return
        if not self._hook_plan:
            self._build_hook_plan()

        for name, info in self._hook_plan.items():
            module = info["module"]
            metrics = info["metrics"]
            handle = module.register_forward_hook(self._hook_fn(name, metrics))
            self._hook_handles.append(handle)
            metric_list = ", ".join(m.name for m in metrics)
            logger.debug("[hook registered] layer='{}' -> metrics=[{}]", name, metric_list)

        logger.info("Hook registration completed, total {} hooks", len(self._hook_handles))

    def _hook_fn(self, layer_name, metrics):
        def hook(module, inputs, outputs, layer=layer_name, metrics_bound=metrics):
            ctx = RuntimeContext.get()
            hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs
            extra = dict(ctx.kwargs) if ctx else {}
            extra["model"] = self.model
            extra["labels"] = ctx.labels
            for metric in metrics_bound:
                self.collector.update(metric, layer, hidden_states, **extra)
                logger.trace(
                    "[hook trigger] metric='{}' layer='{}'",
                    metric.name,
                    layer,
                )
        return hook

    def _clear_hooks(self):
        removed = len(self._hook_handles)
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        if removed:
            logger.info("Removed {} hooks.", removed)

    def reset_hooks(self):
        logger.info("Resetting hooks ...")
        self._clear_hooks()
        self._hook_plan = {}
        self._register_hooks()

    def forward(self, *args, **kwargs):
        if "labels" not in kwargs and "input_ids" in kwargs:
            kwargs["labels"] = kwargs["input_ids"].clone()
            logger.info("Auto-generated labels from input_ids")

        if not self._hook_handles:
            logger.warning(
                "No active hooks detected before forward. "
                "If you modified metrics or model modules, please call `reset_hooks()`."
            )

        logger.debug("Model forward started.")
        RuntimeContext.start(self.model, **kwargs)

        try:
            out = self.model(*args, **kwargs)
        finally:
            RuntimeContext.clear()
            logger.debug("Model forward completed.")

        return out

    def collect(self):
        metrics = self.collector.collect()
        self._summary(metrics)
        failed_metrics = [
            name for name, s in self.collector.status.items() if s["error_flag"]
        ]
        if failed_metrics:
            logger.warning("Some metrics failed: {}", failed_metrics)
            logger.warning("See detailed trace in deepct_runtime.log.")
        return metrics

    def _summary(self, metrics):
        logger.info("Generating DeepCT summary ...")
        summary(metrics)

    @staticmethod
    def list_metrics():
        return list_registered_metrics()