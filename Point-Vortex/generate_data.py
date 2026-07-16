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
    snr: int | None = None,
):
    num_nodes = adjacency.size(0)
    device = adjacency.device
    dtype = adjacency.dtype

    adjacency = adjacency.clone()
    adjacency.fill_diagonal_(0.0)

    states = torch.zeros(num_samples, num_nodes, 2, device=device, dtype=dtype)
    derivatives = torch.zeros_like(states)

    states[0, :, 0].uniform_(-5.0, 5.0)
    states[0, :, 1].uniform_(-5.0, 5.0)

    dt = 0.01
    coeff = 1.0 / (2.0 * torch.pi)

    circulations = torch.ones(num_nodes, device=device, dtype=dtype)

    # adjacency[alpha, beta] means beta influences alpha
    interaction_weights = adjacency * circulations.unsqueeze(0)

    def rhs(state: torch.Tensor) -> torch.Tensor:
        x = state[:, 0]
        y = state[:, 1]


        delta_x = x[:, None] - x[None, :]
        delta_y = y[:, None] - y[None, :]

        r_squared = delta_x.square() + delta_y.square()
        r_squared.fill_diagonal_(1.0)

        velocity_x_pairwise = (
            -coeff
            * interaction_weights
            * delta_y
            / r_squared
        )

        velocity_y_pairwise = (
            coeff
            * interaction_weights
            * delta_x
            / r_squared
        )

        velocity_x_pairwise.fill_diagonal_(0.0)
        velocity_y_pairwise.fill_diagonal_(0.0)

        velocity_x = velocity_x_pairwise.sum(dim=1)
        velocity_y = velocity_y_pairwise.sum(dim=1)

        return torch.stack((velocity_x, velocity_y), dim=-1)

    for t in range(num_samples):
        derivatives[t] = rhs(states[t])

        if t < num_samples - 1:
            states[t + 1] = states[t] + dt * derivatives[t]

    if snr is not None:
        derivatives = add_gaussian_noise_db(derivatives, snr)

    return states.cpu(), derivatives.cpu()