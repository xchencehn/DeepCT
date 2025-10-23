class DeepCTCollector:
    def __init__(self, metrics):
        self.metrics = metrics

    def collect(self, layer_name, hidden_states):
        for m in self.metrics:
            m.update(layer_name, hidden_states)

    def compute_all(self):
        return {m.name: m.compute() for m in self.metrics}