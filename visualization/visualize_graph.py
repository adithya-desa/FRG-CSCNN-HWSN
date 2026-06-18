import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from math import sqrt
from matplotlib.lines import Line2D

# -----------------------------
# PARAMETERS
# -----------------------------
DATA_PATH = "data/data/sensor_data.npy"
DISTANCE_THRESHOLD = 30          # communication range
CLUSTER_HEAD_ENERGY = 0.8        # threshold for cluster head

# -----------------------------
# LOAD SENSOR DATA
# -----------------------------
sensor_data = np.load(DATA_PATH)

positions = sensor_data[:, 0:2]
energy = sensor_data[:, 2]
num_nodes = sensor_data.shape[0]

# Identify cluster heads
cluster_heads = energy >= CLUSTER_HEAD_ENERGY

# -----------------------------
# BUILD GRAPH
# -----------------------------
G = nx.Graph()

for i in range(num_nodes):
    G.add_node(i, pos=(positions[i][0], positions[i][1]))

for i in range(num_nodes):
    for j in range(i + 1, num_nodes):
        dist = sqrt(
            (positions[i][0] - positions[j][0]) ** 2 +
            (positions[i][1] - positions[j][1]) ** 2
        )
        if dist <= DISTANCE_THRESHOLD:
            G.add_edge(i, j)

pos = nx.get_node_attributes(G, 'pos')

# -----------------------------
# VISUALIZATION
# -----------------------------
plt.figure(figsize=(12, 8))

# Normal nodes
normal_nodes = [i for i in range(num_nodes) if not cluster_heads[i]]
cluster_nodes = [i for i in range(num_nodes) if cluster_heads[i]]

nodes = nx.draw_networkx_nodes(
    G,
    pos,
    nodelist=normal_nodes,
    node_size=120,
    node_color=energy[normal_nodes],
    cmap=plt.cm.viridis
)

# Cluster heads (drawn separately)
nx.draw_networkx_nodes(
    G,
    pos,
    nodelist=cluster_nodes,
    node_size=250,
    node_color="red",
    node_shape="^"
)

nx.draw_networkx_edges(G, pos, alpha=0.4)

# -----------------------------
# COLOR BAR (ENERGY SCALE)
# -----------------------------
cbar = plt.colorbar(nodes, shrink=0.75)
cbar.set_label("Energy Level", rotation=270, labelpad=15)

# -----------------------------
# LEGEND (CLUSTER INFO)
# -----------------------------
legend_elements = [
    Line2D([0], [0], marker='o', color='w',
           label='Normal Sensor Node',
           markerfacecolor='green', markersize=8),

    Line2D([0], [0], marker='^', color='w',
           label='Cluster Head (High Energy)',
           markerfacecolor='red', markersize=10)
]

plt.legend(
    handles=legend_elements,
    loc='upper right',
    title="Node Types"
)

# -----------------------------
# EXPLANATION BOX (RIGHT SIDE)
# -----------------------------
textstr = (
    "Energy Color Mapping:\n"
    "• Dark Blue → Low Energy\n"
    "• Green → Medium Energy\n"
    "• Yellow → High Energy\n\n"
    "Cluster Head Criteria:\n"
    f"• Energy ≥ {CLUSTER_HEAD_ENERGY}\n\n"
    "Edges:\n"
    "• Communication links\n"
    "• Distance ≤ threshold"
)

plt.gcf().text(
    0.83, 0.5, textstr,     # move outside plot
    fontsize=10,
    va="center",
    bbox=dict(boxstyle="round", facecolor="whitesmoke")
)


# -----------------------------
# FINAL TOUCHES
# -----------------------------
plt.title("HWSN Graph from sensor_data.npy\n(Energy Levels & Cluster Heads)")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.grid(True)
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.show()
