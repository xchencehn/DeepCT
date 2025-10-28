import time

import torch


class Collector:

    def __init__(self, metrics):
        self.metrics = metrics
        self.status = {
            m.name: {
                "iter": 0,
                "time_record": [],
                "error": [],
                "error_m": False
            } for m in metrics
        }

    def update(self, layer_name, hidden_states):
        for m in self.metrics:
            torch.cuda.synchronize()
            start_time = time.time()
            try:
                m.update(layer_name, hidden_states)
                torch.cuda.synchronize()
                end_time = time.time()
            except Exception as e:
                torch.cuda.synchronize()
                end_time = time.time()
                error_info = {
                    'message': str(e),
                    'exception_type': type(e).__name__,
                    'layer_name': layer_name,
                    'metric_name': m.name,
                    'hidden_states_shape': getattr(hidden_states, 'shape', 'Unknown'),
                    'execution_time': end_time - start_time,
                    'timestamp': time.time()
                }
                self.status[m.name]["error"].append(error_info)
                self.status[m.name]["error_m"] = True

            self.status[m.name]["iter"] += 1
            self.status[m.name]["time_record"].append(end_time - start_time)

    def collect(self):
        print(f"Collecting {len(self.metrics)} metrics, "
              f"Success {len(self.metrics) - sum([self.status[m.name]['error_m'] for m in self.metrics])}")
        return {m.name: m.compute() for m in self.metrics}
