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
    coupling: float = 0.15 # Kept to avoid breaking signature
):
    num_nodes = adjacency.size(0)
    device = adjacency.device

    # Initialize all 3 dimensions cleanly
    states = torch.zeros(num_samples, num_nodes, 3, device=device)
    derivatives = torch.zeros_like(states)

    # Initial positions
    states[0, :, 0].uniform_(-5.0, 5.0)
    states[0, :, 1].uniform_(-5.0, 5.0)

    dt = 0.01
    gamma = 1.0
    coeff = 1.0 / (2.0 * torch.pi)

    for t in range(num_samples):
        x = states[t, :, 0]
        y = states[t, :, 1]

        # 2D Distances: Shape (num_nodes, num_nodes)
        dx_mat = x.unsqueeze(1) - x.unsqueeze(0) # x_j - x_i
        dy_mat = y.unsqueeze(1) - y.unsqueeze(0) # y_j - y_i
        
        r_sq = dx_mat**2 + dy_mat**2
        r_sq = r_sq + torch.eye(num_nodes, device=device) * 1e-8

        # Standard Point Vortex Velocity Equations:
        # dx_i/dt = -1/(2pi) * sum( gamma_j * (y_i - y_j) / r_ij^2 )
        # dy_i/dt =  1/(2pi) * sum( gamma_j * (x_i - x_j) / r_ij^2 )
        
        x_pairwise = (-dy_mat) / r_sq
        y_pairwise = (-dx_mat) / r_sq

        # Zero out self-interaction explicitly 
        x_pairwise.fill_diagonal_(0.0)
        y_pairwise.fill_diagonal_(0.0)

        # Apply network adjacency matrix constraints
        dx = -coeff * gamma * (x_pairwise * adjacency).sum(dim=1)
        dy = coeff * gamma * (y_pairwise * adjacency).sum(dim=1)

        derivatives[t, :, 0] = dx
        derivatives[t, :, 1] = dy

        if t < num_samples - 1:
            states[t + 1, :, 0] = x + dt * dx
            states[t + 1, :, 1] = y + dt * dy

    if snr is not None:
        derivatives = add_gaussian_noise_db(derivatives, snr)

    return states.cpu(), derivatives.cpu()