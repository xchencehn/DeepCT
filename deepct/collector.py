import time
import torch
from .tools import logger, log_exception, log_timing

class Collector:

    def __init__(self, metrics):
        self.metrics = metrics
        self.status = {
            m.name: {
                "iter": 0,
                "time_record": [],
                "error": [],
                "error_flag": False,
            } for m in metrics
        }

    def update(self, layer_name, hidden_states, **kwargs):
        for m in self.metrics:
            start = time.time()
            try:
                m.update(layer_name, hidden_states, **kwargs)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                elapsed = time.time() - start
                log_timing(m.name, layer_name, elapsed)

            except Exception as e:
                elapsed = time.time() - start
                log_exception(e, layer=layer_name, metric=m.name)
                self.status[m.name]["error"].append({
                    "exception": type(e).__name__,
                    "message": str(e),
                    "layer": layer_name,
                    "time": elapsed
                })
                self.status[m.name]["error_flag"] = True

            finally:
                self.status[m.name]["iter"] += 1
                self.status[m.name]["time_record"].append(elapsed)

    def collect(self):
        total = len(self.metrics)
        failed = sum(self.status[m.name]["error_flag"] for m in self.metrics)
        logger.success(f"Collecting {total} metrics : success={total - failed}, failed={failed}")
        return {m.name: m.compute() for m in self.metrics}