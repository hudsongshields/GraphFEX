
import torch

def random_undirected_adjacency(num_nodes: int, target_degree: float):
    probability = target_degree / (num_nodes - 1)

    # Generate only upper-triangular edges
    rand_mat = torch.rand(num_nodes, num_nodes)
    upper = torch.triu(
        (rand_mat < probability).float(),
        diagonal=1
    )

    # Mirror to make network undirected
    adjacency = upper + upper.T

    # Ensure no isolated nodes
    for node in range(num_nodes):
        if adjacency[node].sum() == 0:
            neighbor = (node + 1) % num_nodes
            adjacency[node, neighbor] = 1.0
            adjacency[neighbor, node] = 1.0

    return adjacency