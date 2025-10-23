import json
import datetime
import pandas as pd
import torch

class Summary:
    def __init__(self, metrics, hook_log=None, runtime_info=None):
        """
        metrics: list[Metric] - 已注册指标对象
        hook_log: list[dict]   - Hook 注册日志
        runtime_info: dict     - DeepCT 运行信息
        """
        self.metrics = metrics
        self.hook_log = hook_log or []
        self.runtime_info = runtime_info or {}

    def show(self):
        lines = []
        lines.append("=== DeepCT Framework Summary ===\n")

        # 框架版本 & 环境
        lines.append("[Runtime Info]")
        for k, v in self.runtime_info.items():
            lines.append(f"  {k}: {v}")

        # 指标注册信息
        lines.append("\n[Registered Metrics]")
        for m in self.metrics:
            tl = m.target_layers
            tl_repr = tl if not callable(tl) else "callable_fn(...)"
            lines.append(f"  - {m.name:<15}  target_layers={tl_repr}")

        # Hook 概览
        lines.append("\n[Hook Summary]")
        lines.append(f"  Total hooks: {len(self.hook_log)}")
        for i, h in enumerate(self.hook_log):
            lines.append(f"  {i+1:>2d}. {h['metric']:<12} -> {h['layer']}")

        lines.append("\n===============================\n")
        print("\n".join(lines))

    def save(self, path):
        data = {
            "metrics": [m.name for m in self.metrics],
            "hooks": self.hook_log,
            "runtime": self.runtime_info,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return path