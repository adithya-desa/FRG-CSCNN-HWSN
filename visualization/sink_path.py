import numpy as np

def generate_sink_path(sensor_data):

    positions = sensor_data[:, :2]
    energy = sensor_data[:, 2]

    sorted_nodes = np.argsort(-energy)

    path = []

    for idx in sorted_nodes[:10]:
        path.append(positions[idx])

    return np.array(path)