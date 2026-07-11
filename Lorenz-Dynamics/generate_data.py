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


def make_data(
    num_samples: int,
    adjacency: torch.Tensor,
    snr: int = None,
    coupling=0.15
):
    num_nodes = adjacency.size(0)
    device = adjacency.device

    states = torch.empty(num_samples, num_nodes, 3, device=device)
    derivatives = torch.empty_like(states)

    states[0, :, 0].uniform_(-15.0, 15.0)
    states[0, :, 1].uniform_(-15.0, 15.0)
    states[0, :, 2].uniform_(5.0, 35.0)

    dt = 0.01
    sigma = 10.0
    rho = 28.0
    b = 8.0 / 3.0
    coupling_strength = coupling

    for t in range(num_samples):
        x_i = states[t, :, 0]
        y_i = states[t, :, 1]
        z_i = states[t, :, 2]

        self_dynamics = sigma * (y_i - x_i)

        # pairwise[i, j] = x_j - x_i
        pairwise = x_i.unsqueeze(0) - x_i.unsqueeze(1)

        # sum_j A_ij * (x_j - x_i)
        coupling = coupling_strength * (pairwise * adjacency).sum(dim=1)

        dx = self_dynamics + coupling
        dy = x_i * (rho - z_i) - y_i
        dz = x_i * y_i - b * z_i

        derivatives[t, :, 0] = dx
        derivatives[t, :, 1] = dy
        derivatives[t, :, 2] = dz

        if t < num_samples - 1:
            states[t + 1, :, 0] = states[t, :, 0] + dt * dx
            states[t + 1, :, 1] = states[t, :, 1] + dt * dy
            states[t + 1, :, 2] = states[t, :, 2] + dt * dz

    if snr is not None:
        derivatives = add_gaussian_noise_db(derivatives, snr)

    return states.cpu(), derivatives.cpu()