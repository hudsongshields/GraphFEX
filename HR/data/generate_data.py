from FEX.utils import numerical_deriv
import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt



def make_adjacency(num_nodes: int, probability: float, device) -> torch.Tensor:
    adjacency = (torch.rand(num_nodes, num_nodes) < probability).float()
    adjacency.fill_diagonal_(0.0)
    for node in range(num_nodes):
        if adjacency[node].sum() == 0:
            adjacency[node, (node + 1) % num_nodes] = 1.0
    return adjacency.cpu()

#scale free
def make_static_sf_adjacency(num_nodes: int, num_edges: int, gamma_in: float = 5.0, gamma_out: float = 5.0) -> torch.Tensor:
    """
    Vectorized PyTorch implementation of the Goh-Kahng-Kim Static SF Model 
    (Optimized from Xinjie Zhang's script).
    """
    alpha_in = 1.0 / (gamma_in - 1.0)
    alpha_out = 1.0 / (gamma_out - 1.0)
    
    # Generate power-law fitness weights
    idx = 1.0 + np.arange(num_nodes)
    w_in = 1.0 / (idx ** alpha_in)
    w_out = 1.0 / (idx ** alpha_out)
    
    # Shuffle to decouple in/out correlations
    w_in = np.random.permutation(w_in)
    w_out = np.random.permutation(w_out)
    
    # Normalize into strict probability distributions
    p_in = w_in / np.sum(w_in)
    p_out = w_out / np.sum(w_out)
    
    #  Vectorized edge sampling (Oversample slightly to account for duplicate links)
    oversample_factor = 2
    sampled_sources = np.random.choice(num_nodes, size=num_edges * oversample_factor, p=p_out)
    sampled_targets = np.random.choice(num_nodes, size=num_edges * oversample_factor, p=p_in)
    
    # Filter out self-loops and parallel duplicates using a set
    unique_edges = set()
    for s, t in zip(sampled_sources, sampled_targets):
        if s != t: # Prevent self-loops
            unique_edges.add((t, s)) # Target, Source for matrix notation
        if len(unique_edges) >= num_edges:
            break
            
    adjacency = torch.zeros(num_nodes, num_nodes)
    if unique_edges:
        edges_idx = torch.tensor(list(unique_edges), dtype=torch.long)
        adjacency[edges_idx[:, 0], edges_idx[:, 1]] = 1.0
        
    return adjacency


def add_gaussian_noise_db(data: torch.Tensor, snr_db: float):

    column_variances = torch.var(data, dim=0, correction=1)

    signal_power = column_variances.mean()
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise_std = torch.sqrt(noise_power)

    noise = noise_std * torch.randn_like(data)
    return data + noise


def hindmarsh_rose_rhs(state: torch.Tensor, adjacency: torch.Tensor):
    x = state[:, 0]
    y = state[:, 1]
    z = state[:, 2]

    interaction = 0.15 * (2.0 - x) * (adjacency @ torch.sigmoid(x))

    dx = y - x.pow(3) + 3.0 * x.pow(2) - z + 3.24 + interaction
    dy = 1.0 - 5.0 * x.pow(2) - y
    dz = 0.005 * (4.0 * (x + 1.6) - z)

    return torch.stack((dx, dy, dz), dim=-1)


def rk4_step(state: torch.Tensor, adjacency: torch.Tensor, dt: float):
    k1 = hindmarsh_rose_rhs(state, adjacency)
    k2 = hindmarsh_rose_rhs(state + 0.5 * dt * k1, adjacency)
    k3 = hindmarsh_rose_rhs(state + 0.5 * dt * k2, adjacency)
    k4 = hindmarsh_rose_rhs(state + dt * k3, adjacency)
    return state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def make_timeseries(num_samples: int, adjacency: torch.Tensor, snr: float | None = None, dt: float = 0.01) -> tuple[torch.Tensor, torch.Tensor]:
    num_nodes = adjacency.size(0)
    states = torch.empty(num_samples, num_nodes, 3, device=adjacency.device, dtype=adjacency.dtype)
    states[0, :, 0].uniform_(-2.0, 2.0)
    states[0, :, 1].uniform_(-8.0, 4.0)
    states[0, :, 2].uniform_(0.0, 5.0)

    with torch.no_grad():
        for t in range(1, num_samples):
            states[t] = rk4_step(states[t - 1], adjacency, dt)

    observed_states = states.clone()
    if snr is not None:
        observed_states = add_gaussian_noise_db(observed_states, snr)
    observed_derivatives = numerical_deriv.five_point(observed_states, dt=dt)

    # Five-point differentiation estimates derivatives at indices 2:-2.
    observed_states = observed_states[2:-2]

    return observed_states.cpu(), observed_derivatives.cpu()
