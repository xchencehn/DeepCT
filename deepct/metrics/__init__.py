from .registry import _METRIC_REGISTRY, register_metric
from .selfattn_cov_spectrum import SelfAttentionCovarianceSpectrum
from .selfattn_repr_correlation import SelfAttnRepresentationalCorrelation
from .perplexity_metric import PerplexityMetric
from .layerwise_perplexity_metric import LayerwisePerplexityMetric
from .layer_representation_intrinsic_dimension import LayerRepresentationIntrinsicDimension


def get_metric_instance(name):
    """
    Instantiate the corresponding Metric class by its name.
    """
    if name not in _METRIC_REGISTRY:
        available = list(_METRIC_REGISTRY.keys())
        suggestion = ""
        if available:
            suggestion = f"\n→ Available metrics: {available}"
        raise ValueError(
            f"[DeepCT] ❌ Unknown metric '{name}'. "
            f"Did you spell it correctly?{suggestion}"
        )
    return _METRIC_REGISTRY[name]()


def list_registered_metrics():
    """
    List all registered Metric names with their brief descriptions.
    """
    metrics_info = {}
    for name, cls in _METRIC_REGISTRY.items():
        doc = (cls.__doc__ or "").strip().split("\n")[2]
        metrics_info[name] = doc if doc else "(no description)"
    print("=== Registered DeepCT Metrics ===")
    for k, v in metrics_info.items():
        print(f"{k:30s} : {v}")
    print("====================================")
    return metrics_info


__all__ = [
    "get_metric_instance",
    "register_metric",
    "list_registered_metrics",
]
