import numpy as np
import os

def generate_sensor_data(
    num_nodes=30,
    area_size=100
):
    positions = np.random.rand(num_nodes, 2) * area_size
    energy = np.random.uniform(0.5, 1.0, (num_nodes, 1))
    node_type = np.random.randint(0, 2, (num_nodes, 1))

    data = np.hstack([positions, energy, node_type])
    return data

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    sensor_data = generate_sensor_data()
    np.save("data/sensor_data.npy", sensor_data)
    print("sensor_data.npy created successfully!")
    