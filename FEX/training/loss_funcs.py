
from typing import List, Optional
from ..models.learnable_tree import FEX
import torch
import torch.nn.functional as F

# the easy loop way
def test_total_loss(batch_x, batch_dy_val, self_tree, inter_tree, adj_matrix):
    device = batch_x.device
    B, G, D = batch_x.shape

    # forcing: run each node independently
    group_inputs = batch_x.reshape(B * G, D)
    forcing_out = self_tree(group_inputs).reshape(B, G, 1)  # [B, G, 1]
    forcing_out = forcing_out.squeeze(-1) # [B, G]

    # interaction: explicit loop, no scatter
    interaction_out = torch.zeros(B, G, device=device)
    for i in range(G):
        neighbors = adj_matrix[i].nonzero(as_tuple=True)[0]
        for j in neighbors:
            inter_input = torch.cat([batch_x[:, i, :], batch_x[:, j, :]], dim=-1)  # [B, 6]
            interaction_out[:, i] += adj_matrix[i, j] * inter_tree(inter_input).reshape(B)

    pred = forcing_out # + interaction_out
    
    loss = F.mse_loss(pred, batch_dy_val[:, :, 0])
    return loss

def total_loss(batch_x, batch_dy_val, self_tree, inter_tree=None, adj_mat_nodes=None, adj_mat_edges=None, scatter_idx=None, norm_coeff=1.0):

    B = batch_x.size(0)
    G = batch_x.size(1)
    group_indices = torch.arange(G, device=batch_x.device)

    # Batched self interactions
    self_tree = self_tree.to(batch_x.device)
    inter_tree = inter_tree.to(batch_x.device) if inter_tree is not None else None
    group_inputs = batch_x[:, group_indices, :].reshape(B * G, -1).to(batch_x.device) # input nodes across batch and group of nodes
    forcing_out = self_tree(group_inputs)
    forcing_out = forcing_out.reshape(B, G, 1)
    
    if inter_tree is not None:
        num_edges = adj_mat_edges.size(0)

        inter_sources = batch_x[:, adj_mat_nodes, :]
        inter_edges = batch_x[:, adj_mat_edges, :]
        edge_inputs = torch.cat([inter_sources, inter_edges], dim=-1)

        inter_out = inter_tree(edge_inputs.reshape(B * num_edges, -1))
        inter_out = inter_out.reshape(B, num_edges, 1)

        local_idx = scatter_idx.unsqueeze(0).unsqueeze(-1).expand(B, num_edges, 1)
        interaction_out = torch.zeros(B, G, 1, device=batch_x.device)
        interaction_out.scatter_add_(1, local_idx, inter_out)

        batch_dy = forcing_out + (interaction_out * norm_coeff)
    else:
        batch_dy = forcing_out
        
    loss = F.mse_loss(batch_dy, batch_dy_val[:, :, :])

    if not torch.isfinite(batch_dy).all():
        loss = torch.tensor(float('inf'), device=batch_x.device)

    return loss

# with this approach, we may group nodes together, and compute only the interactions of the nodes within that group
def group_loss(batch_x, batch_dy_val, self_tree, inter_tree, adj_mat_nodes, adj_mat_edges, scatter_idx, num_groups):
    group_indices = torch.randperm(batch_x.size(1), device=batch_x.device)
    group_indices = group_indices.chunk(num_groups)

    normalization_coeff = (batch_x.size(1) - 1) / (num_groups - 1)

    total_loss = 0.0
    for group in group_indices:
        group_x = batch_x[:, group, :]
        group_dy_val = batch_dy_val[:, group, :]
        group_adj_nodes = adj_mat_nodes[group]
        group_adj_edges = adj_mat_edges[group]
        group_scatter_idx = scatter_idx[group]

        total_loss += total_loss(group_x, group_dy_val, self_tree, inter_tree, group_adj_nodes, group_adj_edges, group_scatter_idx, norm_coeff=normalization_coeff)
    return total_loss