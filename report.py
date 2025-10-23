import json
import pandas as pd

class DeepCTReport:
    def __init__(self, data):
        self.data = data

    def show(self):
        print("=== DeepCT Report ===")
        for k, v in self.data.items():
            print(f"\n[{k}]")
            for layer, val in v.items():
                print(f"  {layer}: {val:.4f}" if isinstance(val, (int, float)) else f"  {layer}: {val}")
        print("=====================\n")

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.data, f, indent=2)

    def to_dataframe(self):
        return pd.DataFrame(self.data)