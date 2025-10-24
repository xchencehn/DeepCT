# deepct/logger.py
from loguru import logger
import sys
import os
import datetime
import torch

logger.remove()

log_format = (
    "<green>{time:HH:mm:ss}</green> | "
    "<level>{level: <7}</level> | "
    "<cyan>{name}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

logger.add(sys.stdout, level="INFO", colorize=True, format=log_format)

def print_banner(model=None, metrics=None, version="0.1.0"):
    """
    打印 DeepCT 框架启动日志头
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
        "🧠  DeepCT — Deep Computed Tomography for LLMs",
        f"Version     : {version}",
        f"Timestamp   : {timestamp}",
        f"Torch       : {torch_version}",
        f"Device      : {device}",
        f"Model       : {model_name}",
        f"Metrics     : {metric_names}",
        "═" * width,
    ]

    logger.opt(colors=True).info("\n" + "\n".join(banner))
    logger.info("Initializing DeepCT at {}", timestamp)
    logger.info("Loaded model: {}", model_name)
    logger.info("Registered metrics: {}", metric_names)

    return banner


__all__ = ["logger", "print_banner"]