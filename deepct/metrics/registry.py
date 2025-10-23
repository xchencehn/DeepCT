# 注册系统
_METRIC_REGISTRY = {
}

def register_metric():
    def _reg(cls):
        metric_name = cls.name
        _METRIC_REGISTRY[metric_name] = cls
        return cls
    return _reg