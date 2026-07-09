import torch

def make_adjacency(num_nodes: int, probability: float, device) -> torch.Tensor:
    adjacency = (torch.rand(num_nodes, num_nodes) < probability).float()
    adjacency.fill_diagonal_(0.0)
    for node in range(num_nodes):
        if adjacency[node].sum() == 0:
            adjacency[node, (node + 1) % num_nodes] = 1.0
    return adjacency.cpu()


def add_gaussian_noise_db(data: torch.Tensor, snr_db: float):

    column_variances = torch.var(data, dim=0, correction=1)

    signal_power = column_variances.mean()
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise_std = torch.sqrt(noise_power)

    noise = noise_std * torch.randn_like(data)
    return data + noise


def make_data(num_samples: int, adjacency: torch.Tensor, snr: int = None) -> tuple[torch.Tensor, torch.Tensor]:
    num_nodes = adjacency.size(0)
    states = torch.empty(num_samples, num_nodes, 3, device=adjacency.device)
    states[..., 0].uniform_(-10.0, 10.0)
    states[..., 1].uniform_(-10.0, 10.0)
    states[..., 2].uniform_(-2.0, 4.0)

    x_i = states[..., 0]
    y_i = states[..., 1]
    z_i = states[..., 2]
    omega = 1 + 0.1 * torch.randn(num_nodes, device=adjacency.device)

    self_dynamics = -omega * y_i - z_i
    pairwise = 0.15 * (x_i.unsqueeze(1) - x_i.unsqueeze(2))
    
    dx = self_dynamics + (pairwise * adjacency.unsqueeze(0)).sum(dim=2)
    dy = omega * x_i + 0.2 * y_i
    dz = 0.2 + z_i * (x_i - 5.7)

    derivatives = torch.zeros_like(states)
    derivatives[..., 0] = dx
    derivatives[..., 1] = dy
    derivatives[..., 2] = dz
    if snr is not None:
        derivatives = add_gaussian_noise_db(derivatives, snr)

    return states.cpu(), derivatives.cpu()