import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import flwr as fl
import torch
import numpy as np
from models.clifford_cnn import CliffordSteerableCNN

print("🌐 Client script started")

class SensorClient(fl.client.NumPyClient):
    def __init__(self):
        print("🧠 Initializing model")
        self.model = CliffordSteerableCNN(in_features=4)

    def get_parameters(self, config=None):
        print("📤 Sending parameters to server")
        return [val.detach().cpu().numpy() for val in self.model.parameters()]

    def set_parameters(self, parameters):
        print("📥 Received parameters from server")
        for param, new in zip(self.model.parameters(), parameters):
            param.data = torch.tensor(new)

    def fit(self, parameters, config):
        print("🏋️ Local training started")
        self.set_parameters(parameters)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        dummy_data = torch.rand((10, 4))

        for _ in range(3):
            optimizer.zero_grad()
            loss = self.model(dummy_data).mean()
            loss.backward()
            optimizer.step()

        print("✅ Local training finished")
        return self.get_parameters(), len(dummy_data), {}

print("🔗 Connecting client to server...")
fl.client.start_numpy_client(
    server_address="localhost:8081",
    client=SensorClient()
)

