import json
import pandas as pd

class Report:
    def __init__(self, data):
        self.data = data

    def show(self):
        print("=== DeepCT Report ===")
        for key, val in self.data.items():
            print(f"\n[{key}]")
            for layer, metric_val in val.items():
                if isinstance(metric_val, (int, float)):
                    print(f"  {layer}: {metric_val:.4f}")
                else:
                    print(f"  {layer}: {metric_val}")
        print("=====================\n")

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def to_dataframe(self):
        return pd.DataFrame(self.data)