from metrics.correlator import CorrelatorMetric
from metrics.intrinsic_dim import IntrinsicDim
from metrics.correlator import CorrelatorMetric

from metrics.registry import _METRIC_REGISTRY

def get_metric_instance(name):
    if name not in _METRIC_REGISTRY:
        raise ValueError(f"Unknown metric '{name}'. Registered metrics: {list(_METRIC_REGISTRY.keys())}")
    return _METRIC_REGISTRY[name]()

__all__ = ["get_metric_instance"]
