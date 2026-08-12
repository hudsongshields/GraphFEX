from FEX.utils import numerical_deriv
import torch

def make_adjacency(num_nodes: int, target_degree: float):
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


def add_gaussian_noise_db(data: torch.Tensor, snr_db: float):

    column_variances = torch.var(data, dim=0, correction=1)

    signal_power = column_variances.mean()
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise_std = torch.sqrt(noise_power)

    noise = noise_std * torch.randn_like(data)
    return data + noise


def lorenz_rhs(state: torch.Tensor, adjacency: torch.Tensor, coupling_strength: float):
    x_i = state[:, 0]
    y_i = state[:, 1]
    z_i = state[:, 2]

    self_dynamics = 10.0 * (y_i - x_i)

    # pairwise[i, j] = x_j - x_i
    pairwise = x_i.unsqueeze(0) - x_i.unsqueeze(1)

    # sum_j A_ij * (x_j - x_i)
    coupling = coupling_strength * (pairwise * adjacency).sum(dim=1)

    dx = self_dynamics + coupling
    dy = x_i * (28.0 - z_i) - y_i
    dz = x_i * y_i - (8.0 / 3.0) * z_i

    return torch.stack((dx, dy, dz), dim=-1)


def rk4_step(state: torch.Tensor, adjacency: torch.Tensor, dt: float, coupling_strength: float):
    k1 = lorenz_rhs(state, adjacency, coupling_strength)
    k2 = lorenz_rhs(state + 0.5 * dt * k1, adjacency, coupling_strength)
    k3 = lorenz_rhs(state + 0.5 * dt * k2, adjacency, coupling_strength)
    k4 = lorenz_rhs(state + dt * k3, adjacency, coupling_strength)
    return state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def make_data(
    num_samples: int,
    adjacency: torch.Tensor,
    snr: int = None,
    coupling=3.6
):
    num_nodes = adjacency.size(0)

    states = torch.empty(num_samples, num_nodes, 3, device=adjacency.device, dtype=adjacency.dtype)

    states[0, :, 0].uniform_(-15.0, 15.0)
    states[0, :, 1].uniform_(-15.0, 15.0)
    states[0, :, 2].uniform_(5.0, 35.0)

    dt = 0.01

    with torch.no_grad():
        for t in range(1, num_samples):
            states[t] = rk4_step(states[t - 1], adjacency, dt, coupling)

    observed_states = states.clone()

    if snr is not None:
        observed_states = add_gaussian_noise_db(observed_states, snr)
    observed_derivatives = numerical_deriv.five_point(observed_states, dt=dt)

    # Five-point differentiation estimates derivatives at indices 2:-2.
    observed_states = observed_states[2:-2]

    return observed_states.cpu(), observed_derivatives.cpu()