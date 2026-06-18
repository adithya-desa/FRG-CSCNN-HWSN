import torch
from torch_geometric.data import Data

def build_graph(features, threshold=30):
    num_nodes = features.shape[0]
    edge_index = []

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                dist = ((features[i, :2] - features[j, :2]) ** 2).sum() ** 0.5
                if dist < threshold:
                    edge_index.append([i, j])

    edge_index = torch.tensor(edge_index).t().contiguous()
    x = torch.tensor(features, dtype=torch.float)

    return Data(x=x, edge_index=edge_index)
