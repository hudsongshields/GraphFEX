from FEX.helpers import numerical_deriv
import torch
import math
from NumericalExperiments.networks import random_undirected_adjacency


def make_adjacency(num_nodes, min_degree, max_degree, device="cpu", dtype=torch.float32):
    adjacency = torch.zeros(
        num_nodes,
        num_nodes,
        device=device,
        dtype=dtype
    )

    for i in range(num_nodes):

        candidates = torch.cat([
            torch.arange(0, i, device=device),
            torch.arange(i + 1, num_nodes, device=device)
        ])

        degree_i = torch.randint(min_degree, max_degree + 1, (1,), device=device).item()
        perm = torch.randperm(num_nodes - 1, device=device)

        neighbors = candidates[perm[:degree_i]]
        adjacency[i, neighbors] = 1.0

    return adjacency


def add_gaussian_noise_db(
    data: torch.Tensor,
    snr_db: float
):
    column_variances = torch.var(
        data,
        dim=0,
        correction=1
    )

    signal_power = column_variances.mean()

    noise_power = (
        signal_power /
        (10.0 ** (snr_db / 10.0))
    )

    noise_std = torch.sqrt(noise_power)

    noise = noise_std * torch.randn_like(data)

    return data + noise


def make_data(
    num_trajectories: int,
    num_timesteps: int,
    adjacency: torch.Tensor,
    snr: int | None = None,
    coupling: float = 0.1,
    b: float = 1.0,
    lambda_: float = 1.0,
    omega_value: float = 1.0,
    smoothing: bool = True
):
    num_nodes = adjacency.size(0)
    device = adjacency.device
    dtype = adjacency.dtype

    adjacency = adjacency.clone()
    adjacency.fill_diagonal_(0.0)

    # [trajectory, time, node, state_dimension]
    states = torch.zeros(num_trajectories, num_timesteps, num_nodes, 2, device=device, dtype=dtype)

    dt = 0.01

    # Same natural frequency for every oscillator
    omega = torch.full(
        (num_nodes,),
        omega_value,
        device=device,
        dtype=dtype
    )

    def rhs(state: torch.Tensor):

        x = state[:, 0]
        y = state[:, 1]

        r2 = x**2 + y**2

        dx_self = (lambda_ * x - omega * y - r2 * x)
        dy_self = (omega * x + lambda_ * y - r2 * y)

        # Pairwise exponential coupling
        x_i = x[:, None]
        x_j = x[None, :]

        interaction = (torch.exp(b * (x_j - x_i)) - 1.0)
        interaction = adjacency * interaction
        coupling_term = interaction.sum(dim=1)
        coupling_term *= coupling

        dx = dx_self + coupling_term
        dy = dy_self

        return torch.stack([dx, dy], dim=-1)

    # Generate  trajectories
    for traj in range(num_trajectories):
        phase = torch.empty(
            num_nodes,
            device=device,
            dtype=dtype
        ).uniform_(-math.pi, math.pi)

        radius = torch.empty(num_nodes, device=device, dtype=dtype).uniform_(0.5, 1.5)

        states[traj, 0, :, 0] = radius * torch.cos(phase)
        states[traj, 0, :, 1] = radius * torch.sin(phase)

        # Integrate trajectory
        for t in range(num_timesteps):
            if t < num_timesteps - 1:

                state = states[traj, t]

                k1 = rhs(state)
                k2 = rhs(state + 0.5 * dt * k1)
                k3 = rhs(state + 0.5 * dt * k2)
                k4 = rhs(state + dt * k3)

                states[traj, t + 1] = (
                    state
                    + (dt / 6.0)
                    * (k1 + 2*k2 + 2*k3 + k4)
                )

    # Add noise after generating all trajectories
    # generate derivatives from observational data (finite difference)
    observed_states = states.clone()
    observed_derivatives = torch.zeros_like(observed_states)

    if snr is not None:
        tmp_flat_states = observed_states.reshape(num_trajectories * num_timesteps, num_nodes, 2)
        observed_states = add_gaussian_noise_db(tmp_flat_states, snr)
        observed_states = observed_states.reshape(num_trajectories, num_timesteps, num_nodes, 2)

    derivatives = []
    tmp_states = []
    for traj in range(num_trajectories):
        if smoothing: 
            observed_state, observed_derivative = numerical_deriv.smoothed_five_point(observed_states[traj], dt=dt)
        else:
            observed_state, observed_derivative = numerical_deriv.five_point(observed_states, dt=dt)
        derivatives.append(observed_derivative)
        tmp_states.append(observed_state)

    observed_derivatives = torch.stack(derivatives, dim=0)
    observed_states = torch.stack(tmp_states, dim=0)

    # Flatten trajectory
    states_flat = observed_states.reshape(num_trajectories * (num_timesteps - 4), num_nodes, 2)
    derivatives_flat = observed_derivatives.reshape(num_trajectories * (num_timesteps - 4), num_nodes, 2)

    return (
        states_flat.cpu(),
        derivatives_flat.cpu(),
        omega.cpu()
    )