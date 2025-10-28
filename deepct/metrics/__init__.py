from .registry import _METRIC_REGISTRY, register_metric
from .selfattn_cov_spectrum import SelfAttentionCovarianceSpectrum
from .selfattn_repr_correlation import SelfAttnRepresentationalCorrelation
from .perplexity_metric import PerplexityMetric

def get_metric_instance(name):
    if name not in _METRIC_REGISTRY:
        raise ValueError(f"Unknown metric '{name}'. Available: {list(_METRIC_REGISTRY.keys())}")
    return _METRIC_REGISTRY[name]()

__all__ = ["get_metric_instance", "register_metric"]