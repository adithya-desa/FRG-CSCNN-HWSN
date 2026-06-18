import numpy as np

def plan_sink_path(scores, positions):
    """
    Selects the next sink position based on node-level scores
    """

    num_nodes = positions.shape[0]

    # Ensure scores are node-level
    scores = np.array(scores).flatten()

    # Truncate or reshape scores safely
    if scores.shape[0] != num_nodes:
        scores = scores[:num_nodes]

    best_node = np.argmax(scores)

    return positions[best_node]
