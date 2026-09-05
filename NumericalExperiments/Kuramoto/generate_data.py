import torch
import math
from NumericalExperiments.networks import random_undirected_adjacency

make_adjacency = random_undirected_adjacency

def make_normalized_adjacency(num_nodes, degree):
    adjacency = make_adjacency(num_nodes, degree)
    degrees_per_node = adjacency.sum(dim=1, keepdim=True)
    normalized_adjacency = torch.where(degrees_per_node > 0, adjacency / degrees_per_node, torch.zeros_like(adjacency))
    return normalized_adjacency

def make_arni_adjacency(num_nodes, degree, device="cpu", dtype=torch.float32):
    adjacency = torch.zeros(
        num_nodes, num_nodes,
        device=device,
        dtype=dtype
    )

    for i in range(num_nodes):
        # All possible incoming neighbors except self
        candidates = torch.cat([
            torch.arange(0, i, device=device),
            torch.arange(i + 1, num_nodes, device=device)
        ])

        # Pick exactly `degree` incoming neighbors
        perm = torch.randperm(num_nodes - 1, device=device)
        neighbors = candidates[perm[:degree]]

        weights = torch.ones(degree, device=device, dtype=dtype) / degree

        adjacency[i, neighbors] = weights

    return adjacency


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
    coupling: float = 1.0,
):
    num_nodes = adjacency.size(0)
    device = adjacency.device
    dtype = adjacency.dtype

    adjacency = adjacency.clone()
    adjacency.fill_diagonal_(0.0)

    # [time, node, state_dimension]
    states = torch.zeros(num_samples, num_nodes, 1, device=device, dtype=dtype)
    derivatives = torch.zeros_like(states)

    # Initial oscillator phases
    states[0, :, 0].uniform_(-math.pi, math.pi)

    dt = 0.01

    # Natural frequencies omega_i
    omega = -2.0 + 4.0 * torch.rand(num_nodes, device=device, dtype=dtype)

    def rhs(state: torch.Tensor):
        theta = state[:, 0]

        # phase_difference[i,j] = theta_j - theta_i
        phase_difference = theta[None, :] - theta[:, None]

        mode1 = torch.sin(phase_difference - 1.05)
        mode2 = 0.33 * torch.sin(2.0 * phase_difference)

        interaction = adjacency * (mode1 + mode2)
        coupling_term = interaction.sum(dim=1)

        dtheta = omega + coupling * coupling_term

        return dtheta.unsqueeze(-1)

    for t in range(num_samples):
        derivatives[t] = rhs(states[t])

        if t < num_samples - 1:
            state = states[t]

            k1 = rhs(state)
            k2 = rhs(state + 0.5 * dt * k1)
            k3 = rhs(state + 0.5 * dt * k2)
            k4 = rhs(state + dt * k3)

            states[t + 1] = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    # return states to [-pi, pi]
    states = (states + math.pi) % (2 * math.pi) - math.pi
    if snr is not None:
        derivatives = add_gaussian_noise_db(derivatives, snr)

    return states.cpu(), derivatives.cpu()