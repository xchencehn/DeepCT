# 注册系统
_METRIC_REGISTRY = {
}

def register_metric():
    def _reg(cls):
        _METRIC_REGISTRY[cls.name] = cls
        return cls
    return _reg