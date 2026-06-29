import torch

def make_adjacency(num_nodes: int, probability: float, device) -> torch.Tensor:
    adjacency = (torch.rand(num_nodes, num_nodes) < probability).float()
    adjacency.fill_diagonal_(0.0)
    for node in range(num_nodes):
        if adjacency[node].sum() == 0:
            adjacency[node, (node + 1) % num_nodes] = 1.0
    return adjacency.to(device)


def make_data(num_samples: int, adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    num_nodes = adjacency.size(0)
    states = torch.empty(num_samples, num_nodes, 3, device=adjacency.device)
    states[..., 0].uniform_(-2.0, 2.0)
    states[..., 1].uniform_(-8.0, 4.0)
    states[..., 2].uniform_(0.0, 5.0)

    x_i = states[..., 0]
    y_i = states[..., 1]
    z_i = states[..., 2]
    self_dynamics = y_i - x_i.pow(3) + 3.0 * x_i.pow(2) - z_i + 3.24
    pairwise = 0.15 * (2.0 - x_i).unsqueeze(2) * torch.sigmoid(x_i).unsqueeze(1)
    dx = self_dynamics + (pairwise * adjacency.unsqueeze(0)).sum(dim=2)

    dy = 1 - 5 * x_i.pow(2) - y_i
    dz = 0.004 * (4 * (x_i + 1.6) - z_i)

    derivatives = torch.zeros_like(states)
    derivatives[..., 0] = dx
    derivatives[..., 1] = dy
    derivatives[..., 2] = dz
    return states.cpu(), derivatives.cpu()