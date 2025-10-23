# 注册系统
_METRIC_REGISTRY = {
}

def register_metric(name=None):
    def _reg(cls):
        metric_name = name or cls.name
        _METRIC_REGISTRY[metric_name] = cls
        return cls
    return _reg