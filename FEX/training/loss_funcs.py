
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
    B, G, D = batch_x.shape
    forcing_out = self_tree(batch_x.reshape(B * G, D)).reshape(B, G, 1)
    batch_dy = forcing_out

    if inter_tree is not None and adj_mat_nodes.numel() > 0:
        num_edges = adj_mat_nodes.numel()
        inter_sources = batch_x[:, adj_mat_nodes, :]
        inter_edges = batch_x[:, adj_mat_edges, :]
        edge_inputs = torch.cat([inter_sources, inter_edges], dim=-1)
        inter_out = inter_tree(edge_inputs.reshape(B * num_edges, -1)).reshape(B, num_edges, 1)

        local_idx = scatter_idx.view(1, num_edges, 1).expand(B, num_edges, 1)
        interaction_out = torch.zeros(B, G, 1, device=batch_x.device, dtype=forcing_out.dtype)
        interaction_out.scatter_add_(1, local_idx, inter_out)
        batch_dy = forcing_out + interaction_out * norm_coeff

    return F.mse_loss(batch_dy, batch_dy_val)


def group_loss(batch_x, batch_dy_val, self_tree, inter_tree, adj_mat_nodes, adj_mat_edges, scatter_idx, num_groups):
    num_nodes = batch_x.size(1)
    groups = torch.randperm(num_nodes, device=batch_x.device).chunk(num_groups)

    adj_mat_nodes = adj_mat_nodes.to(device=batch_x.device, dtype=torch.long)
    adj_mat_edges = adj_mat_edges.to(device=batch_x.device, dtype=torch.long)
    scatter_idx = scatter_idx.to(device=batch_x.device, dtype=torch.long)

    group_losses = []
    for group in groups:
        group_size = group.numel()
        group_x = batch_x[:, group, :]
        group_dy = batch_dy_val[:, group, :]

        global_to_local = torch.full((num_nodes,), -1, device=batch_x.device, dtype=torch.long)
        global_to_local[group] = torch.arange(group_size, device=batch_x.device)

        edge_mask = (global_to_local[adj_mat_nodes] >= 0) & (global_to_local[adj_mat_edges] >= 0)
        group_adj_nodes = global_to_local[adj_mat_nodes[edge_mask]]
        group_adj_edges = global_to_local[adj_mat_edges[edge_mask]]
        group_scatter_idx = global_to_local[scatter_idx[edge_mask]]

        norm_coeff = (num_nodes - 1) / (group_size - 1) if group_size > 1 else 1.0
        group_losses.append(total_loss(
            group_x, group_dy, self_tree, inter_tree,
            group_adj_nodes, group_adj_edges, group_scatter_idx,
            norm_coeff=norm_coeff
        ))

    return torch.stack(group_losses).mean()