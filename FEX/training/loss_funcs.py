
from typing import List, Optional
from ..models.learnable_tree import FEX
import torch
import torch.nn.functional as F


def euler_loss(
    x_sequential: List[torch.Tensor],
    dt: float,
    forcing_tree: FEX,
    inter_dynam_tree: FEX,
    adj_matrix: torch.Tensor,
    nodes: torch.Tensor,
    edges: torch.Tensor,
    rollout_steps: int = 10,
):
    device = next(forcing_tree.parameters()).device

    # Pick a random run and starting time
    if isinstance(x_sequential, torch.Tensor):
        x_sequential = [x_sequential]
    selected_run = torch.randint(0, len(x_sequential), (1,)).item()
    x_run = x_sequential[selected_run].to(device)
    T, N, D = x_run.shape

    max_start = T - rollout_steps - 1
    if max_start <= 0:
        return torch.tensor(0.0, device=device)
    t_start = torch.randint(0, max_start + 1, (1,)).item()

    # Ground truth window
    x_window = x_run[t_start : t_start + rollout_steps + 1]

    # Initialise rolled-out dim 0 from ground truth at t=0
    x_pred_dim0 = x_window[0, :, 0:1].clone()

    loss = torch.tensor(0.0, device=device)

    for step in range(rollout_steps):
        # current state: predicted dim 0 + known dims
        x_current = torch.cat([x_pred_dim0, x_window[step, :, 1:]], dim=-1)

        # batched forcing
        dx0_forcing = forcing_tree(x_current)

        # batched interaction
        inter_inputs = torch.cat([x_current[nodes], x_current[edges]], dim=-1)
        inter_outputs = inter_dynam_tree(inter_inputs)

        # Scatter-sum interaction contributions per destination node
        dx0_interaction = torch.zeros(N, 1, device=device)
        dx0_interaction.index_add_(0, nodes, inter_outputs)

        dx0 = dx0_forcing + dx0_interaction

        # Euler step (only dim 0)
        x_pred_dim0 = x_pred_dim0 + dt * dx0

        # step loss against ground truth
        x_true_dim0 = x_window[step + 1, :, 0:1]
        loss = loss + F.mse_loss(x_pred_dim0, x_true_dim0)

    return loss / rollout_steps


def group_loss(batch_x, batch_dy_val, group_indices, self_tree, inter_tree, adj_mat_nodes, adj_mat_edges):

        sparse_dx_dt_val = batch_dy_val[:, group_indices, :]

        B = batch_x.size(0)
        G = group_indices.size(0)

        # Batched self interactions
        group_inputs = batch_x[:, group_indices, :].reshape(B * G, -1) # input nodes across batch and group of nodes
        forcing_out = self_tree(group_inputs)
        forcing_out = forcing_out.reshape(B, G, 1)

        # Batched interaction - only consider edges with node indices in the selected group
        aligned_indices = (adj_mat_nodes.unsqueeze(1) == group_indices.unsqueeze(0)).any(dim=1) 
        sparse_nodes = adj_mat_nodes[aligned_indices]
        sparse_edges = adj_mat_edges[aligned_indices]
        num_sparse_edges = sparse_edges.size(0)
        

        # Gather edge features and forward in one call
        inter_sources = batch_x[:, sparse_nodes, :]   # [B, E_g, D]
        inter_edges = batch_x[:, sparse_edges, :]   # [B, E_g, D]
        edge_inputs = torch.cat([inter_sources, inter_edges], dim=-1)  # [B, E_g, 2*D]
        inter_out = inter_tree(edge_inputs.reshape(B * num_sparse_edges, -1))  # [B*E_g, 1]
        inter_out = inter_out.reshape(B, num_sparse_edges, 1)

        # Scatter-sum interaction outputs to group positions
        sparse_indices = (sparse_nodes.unsqueeze(1) == group_indices.unsqueeze(0)).int().argmax(dim=1)
        interaction_out = torch.zeros(B, G, 1, device=batch_x.device)
        local_idx = sparse_indices.unsqueeze(0).unsqueeze(-1).expand(B, num_sparse_edges, 1)
        interaction_out.scatter_add_(1, local_idx, inter_out) # sum contributions from all edges to each node

        batch_dy = forcing_out + interaction_out
        loss = F.huber_loss(batch_dy, sparse_dx_dt_val)

        # print(f"pred derivative: {batch_dy.mean().item():.4f}, true derivative: {sparse_dx_dt_val.mean().item():.4f}, loss: {loss.item():.4f}")

        return loss

def mag_entropy_regularization(fex: FEX):
    params = fex.tree_log_mags()
    log_mags = torch.stack(params)
    mags = F.softplus(log_mags) # recreate magnitude from log_mag
    probs = F.softmax(mags, dim=0)
    entropy = -(probs * torch.log(probs + 1e-8)).sum()  
    return entropy

def mag_reverse_l2_regularization(fex: FEX):
    params = fex.tree_log_mags()
    log_mags = torch.stack(params)
    mags = F.softplus(log_mags) # recreate magnitude from log_mag
    reverse_l2 = - (mags ** 2).sum()
    return reverse_l2