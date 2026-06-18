import numpy as np

class SensorNetwork:
    def __init__(self, num_nodes=50, area=100):
        self.num_nodes = num_nodes
        self.positions = np.random.rand(num_nodes, 2) * area
        self.energy = np.random.uniform(0.5, 1.0, num_nodes)
        self.types = np.random.randint(0, 2, num_nodes)

    def get_features(self):
        return np.hstack([
            self.positions,
            self.energy.reshape(-1, 1),
            self.types.reshape(-1, 1)
        ])
