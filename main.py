from network.sensor_network import SensorNetwork
from network.graph_builder import build_graph
from models.clifford_cnn import CliffordSteerableCNN
from sink.sink_planner import plan_sink_path

import torch

network = SensorNetwork()
features = network.get_features()
graph = build_graph(features)

model = CliffordSteerableCNN(features.shape[1])
predictions = model(torch.tensor(features, dtype=torch.float)).detach().numpy()

next_position = plan_sink_path(predictions, network.positions)

import sys
import subprocess

subprocess.run(
    [sys.executable, "visualization/animate_sink.py"]
)
