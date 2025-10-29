from loguru import logger
import sys
import os
import datetime
import torch
from collections import defaultdict, Counter

_LOG_FILE = "deepct_runtime.log"

def setup_logger(to_file=True):

    logger.remove()

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = "DEBUG"

    fmt_console = (
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan> - "
        "<level>{message}</level>"
    )

    logger.add(sys.stdout, level=log_level, colorize=True, format=fmt_console)

    if to_file:
        fmt_file = (
            "{time:YYYY-MM-DD HH:mm:ss} | "
            "{level: <7} | "
            "{name}:{function}:{line} | "
            "{message}"
        )
        logger.add(
            _LOG_FILE,
            rotation="5 MB",
            level="DEBUG",
            encoding="utf-8",
            format=fmt_file,
            enqueue=True,
        )

    return logger

logger = setup_logger()

# Catch global exceptions (crashes during debugging hook)
def log_exception(exc: Exception, layer=None, metric=None):
    import traceback
    trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    ctx = f"metric={metric or 'unknown'}, layer={layer or 'unknown'}"
    logger.error(f"{ctx}\n{trace}\n")

def log_timing(metric, layer, elapsed):
    logger.debug(f"[timing] {metric} -> {layer} took {elapsed * 1000:.1f} ms")

def banner(model=None, metrics=None, version="0.1.0"):
    """
    Print the startup log header of the DeepCT framework
    """
    width = 55
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        torch_version = torch.__version__
    except Exception:
        torch_version = "unknown"

    device = "unknown"
    if model is not None:
        try:
            device = next(model.parameters()).device
        except Exception:
            device = "unavailable"

    model_name = getattr(getattr(model, "config", None), "_name_or_path", "unknown")
    metric_names = ", ".join([m.name for m in (metrics or [])]) if metrics else "none"

    banner = [
        "═" * width,
        "🚀  DeepCT — Deep Computed Tomography for LLMs",
        f"Version     : {version}",
        f"Timestamp   : {timestamp}",
        f"Torch       : {torch_version}",
        f"Device      : {device}",
        f"Model       : {model_name}",
        f"Metrics     : {metric_names}",
        "═" * width,
    ]

    print("\n".join(banner))
    

    return banner


def summary(metrics):
    print("=" * 60)
    print("DeepCT Metric Summary (using PyTorch)")
    print(f"Metrics count: {len(metrics)}")
    total_components = sum(len(comp_data) for comp_data in metrics.values())
    print(f"Total components: {total_components} (across all metrics)")
    print("=" * 60)

    def parse_path(path):
        parts = path.split('.')
        if len(parts) < 4 or parts[0] != 'model' or parts[1] != 'layers':
            return (-1, 'unknown')
        layer = int(parts[2]) if parts[2].isdigit() else -1
        comp_type = '.'.join(parts[3:])
        return (layer, comp_type)
    
    for metric_name, comp_data in metrics.items():
        print(f"\n[Metric]: {metric_name}")

        data_features = []
        for path, val in comp_data.items():
            if not isinstance(val, torch.Tensor):
                val = torch.tensor(val, dtype=torch.float32)

            feat = {}
            feat["path"] = path
            feat["shape"] = tuple(val.shape)
            feat["type"] = (
                "scalar" if val.numel() == 1
                else "vector" if val.dim() == 1
                else "tensor"
            )

            if val.numel() == 0:
                feat.update({"type": "empty", "norm": 0.0, "mean": 0.0})
            else:
                feat["norm"] = val.norm().item()
                feat["mean"] = val.mean().item()
                feat["value"] = val.item() if val.numel() == 1 else val
            data_features.append(feat)

        type_counts = Counter(f["type"] for f in data_features)
        print("  Data Type Distribution:")
        for t, c in type_counts.items():
            print(f"    {t:>7}: {c} components")

        print("\n  Basic Statistics:")
        scalars = [f["value"] for f in data_features if f["type"] == "scalar"]
        vectors = [f for f in data_features if f["type"] == "vector"]
        tensors = [f for f in data_features if f["type"] == "tensor"]

        # Scalar
        if scalars:
            arr = torch.tensor(scalars, dtype=torch.float32)
            std = arr.std(unbiased=False)
            print(
                f"    Scalar: mean={arr.mean():.6f}, std={std:.6f}, "
                f"min={arr.min():.6f}, max={arr.max():.6f}"
            )
        else:
            print("    Scalar: no valid data")

        # Vector
        if vectors:
            norms = torch.tensor([f["norm"] for f in vectors])
            means = torch.tensor([f["mean"] for f in vectors])
            shapes = [f["shape"] for f in vectors]
            print(f"    Vector: norm=[{norms.min():.6f},{norms.max():.6f}], mean=[{means.min():.6f},{means.max():.6f}], avg_norm={norms.mean():.6f}")

            unique_shapes = sorted(set(shapes))
            if len(unique_shapes) > 6:
                shown = unique_shapes[:3] + ['...'] + unique_shapes[-3:]
            else:
                shown = unique_shapes
            print(f"      Shapes ({len(unique_shapes)} total): {shown}")
        else:
            print("    Vector: no valid data")

        # Tensor
        if tensors:
            norms = torch.tensor([f["norm"] for f in tensors])
            means = torch.tensor([f["mean"] for f in tensors])
            shapes = [f["shape"] for f in tensors]
            print(f"    Tensor: norm=[{norms.min():.6f},{norms.max():.6f}], mean=[{means.min():.6f},{means.max():.6f}], avg_norm={norms.mean():.6f}")

            unique_shapes = sorted(set(shapes))
            if len(unique_shapes) > 6:
                shown = unique_shapes[:3] + ['...'] + unique_shapes[-3:]
            else:
                shown = unique_shapes
            print(f"      Shapes ({len(unique_shapes)} total): {shown}")
        else:
            print("    Tensor: no valid data")

        # Component type aggregation
        comp_type_feats = defaultdict(list)
        for f in data_features:
            _, comp_type = parse_path(f["path"])
            if comp_type != "unknown" and f["type"] != "empty":
                comp_type_feats[comp_type].append(f)
        valid_types = {t: v for t, v in comp_type_feats.items() if len(v) >= 3}
        if valid_types:
            print("\n  Key Component Type Stats:")
            type_repr = []
            for t, feats in valid_types.items():
                vals = torch.tensor([f["norm"] if f["type"] != "scalar" else f["value"] for f in feats])
                type_repr.append((t, vals.mean().item(), len(feats)))
            sorted_types = sorted(type_repr, key=lambda x: x[1], reverse=True)
            top, bottom = sorted_types[:5], sorted_types[-3:] if len(sorted_types) > 5 else []
            for t, v, n in top:
                print(f"    {t:>20}: mean={v:.6f} (n={n})")
            if bottom:
                print("    ...")
                for t, v, n in bottom:
                    print(f"    {t:>20}: mean={v:.6f} (n={n})")
        else:
            print("\n  Key Component Type Stats: not enough data")

        # Layer trends
        layer_feats = defaultdict(list)
        for f in data_features:
            layer, _ = parse_path(f["path"])
            if layer != -1 and f["type"] != "empty":
                layer_feats[layer].append(f)
        layer_repr = []
        for l, feats in layer_feats.items():
            vals = torch.tensor([f["norm"] if f["type"] != "scalar" else f["value"] for f in feats])
            layer_repr.append((l, vals.mean().item()))
        sorted_layers = sorted(layer_repr, key=lambda x: x[0])
        if sorted_layers:
            print("\n  Layer-wise Trend:")
            if len(sorted_layers) > 6:
                show_layers = sorted_layers[:3] + [None] + sorted_layers[-3:]
            else:
                show_layers = sorted_layers
            for item in show_layers:
                if item is None:
                    print("    ...")
                else:
                    l, v = item
                    print(f"    Layer {l:2d}: value={v:.6f}")
            first, last = sorted_layers[0][1], sorted_layers[-1][1]
            trend = "increasing" if last > first * 1.1 else "decreasing" if last < first * 0.9 else "stable"
            print(f"    Overall trend: {trend} (first={first:.6f}, last={last:.6f})")
        else:
            print("\n  Layer-wise Trend: not enough data")

        # Outliers
        all_vals = torch.tensor(
            [f["value"] if f["type"] == "scalar" else f["norm"] for f in data_features if f["type"] != "empty"],
            dtype=torch.float32
        )
        if len(all_vals) > 0:
            arr = all_vals.clone().detach()
            std = arr.std(unbiased=False)
            mean = arr.mean()
            threshold = mean + 3 * std
            outliers = [f for f in data_features if f["type"] != "empty" and
                        (f["value"] if f["type"] == "scalar" else f["norm"]) > threshold.item()]
            if outliers:
                print("\n  Outliers (>3σ):")
                for f in outliers[:5]:
                    layer, comp_type = parse_path(f["path"])
                    val = f["value"] if f["type"] == "scalar" else f["norm"]
                    print(f"    Layer {layer} {comp_type} ({f['type']}, shape={f['shape']}): {val:.6f}")
            else:
                print("\n  Outliers (>3σ): None")
        else:
            print("\n  Outlier Detection: no valid data")

        print("-" * 60)

__all__ = ["logger", "banner", "summary"]