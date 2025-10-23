class Collector:

    def __init__(self, metrics):
        self.metrics = metrics

    def update(self, layer_name, hidden_states):
        for m in self.metrics:
            m.update(layer_name, hidden_states)

    def collect(self):
        return {m.name: m.compute() for m in self.metrics}