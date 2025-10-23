import torch
from collector import DeepCTCollector
from report import DeepCTReport
from metrics import get_metric_instance

def deepct(model, metrics=None, layers="all", verbose=True):
    """
    Wrap a Hugging Face Transformer model for internal metric collection.

    Args:
        model: transformers模型 (AutoModel实例)
        metrics (list[str]): 要收集的指标，例如 ["intrinsic_dim", "layer_corr"]
        layers (str|list): 要分析的层（"all" 或 ["encoder.layer.0", ...]）
        verbose (bool): 是否打印Hook注册信息

    Returns:
        DeepCTWrappedModel 对象，与原模型API一致
    """
    return DeepCTWrappedModel(model, metrics=metrics, layers=layers, verbose=verbose)


class DeepCTWrappedModel(torch.nn.Module):
    def __init__(self, model, metrics, layers="all", verbose=True):
        super().__init__()
        self.model = model
        self.collector = DeepCTCollector([get_metric_instance(m) for m in metrics])
        self.layers = layers
        self.verbose = verbose
        self._register_hooks()

    def _register_hooks(self):
        # for name, module in self.model.named_modules():
        #     if "model.layer" in name or self.layers == "all":
        for name, module in self.model.model.layers.named_children():
            if True:
                module.register_forward_hook(self._hook_fn(name))
                if self.verbose:
                    print(f"[DeepCT] Hook registered on {name}")

    def _hook_fn(self, layer_name):
        def hook(module, inputs, outputs):
            hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs
            self.collector.collect(layer_name, hidden_states)
        return hook

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.model.generate(max_new_tokens=1, *args, **kwargs)

    # def __getattr__(self, name):
    #     return getattr(self.model, name)

    def report(self):
        data = self.collector.compute_all()
        return DeepCTReport(data)