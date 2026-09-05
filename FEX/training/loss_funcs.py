
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


def total_loss(batch_x, batch_dy_val, self_tree, inter_tree=None, adj_mat_nodes=None, adj_mat_edges=None, edge_weights=None):
    B, G, D = batch_x.shape
    forcing_out = self_tree(batch_x.reshape(B * G, D)).reshape(B, G, 1)
    batch_dy = forcing_out

    if inter_tree is not None and adj_mat_nodes.numel() > 0:
        num_edges = adj_mat_nodes.numel()
        inter_sources = batch_x[:, adj_mat_nodes, :]
        inter_edges = batch_x[:, adj_mat_edges, :]
        edge_inputs = torch.cat([inter_sources, inter_edges], dim=-1)
        inter_out = inter_tree(edge_inputs.reshape(B * num_edges, -1)).reshape(B, num_edges, 1) * edge_weights.view(1, num_edges, 1)

        local_idx = adj_mat_nodes.view(1, num_edges, 1).expand(B, num_edges, 1)
        interaction_out = torch.zeros(B, G, 1, device=batch_x.device, dtype=forcing_out.dtype)
        interaction_out.scatter_add_(1, local_idx, inter_out)
        batch_dy = forcing_out + interaction_out

    return F.mse_loss(batch_dy, batch_dy_val)


def group_loss(batch_x, batch_dy_val, self_tree, inter_tree, adj_mat_nodes, adj_mat_edges, edge_weights, num_groups):
    device = batch_x.device
    num_nodes = batch_x.size(1)

    if num_nodes % num_groups != 0:
        raise ValueError("num_nodes must be divisible by num_groups")

    group_size = num_nodes // num_groups

    adj_mat_nodes = adj_mat_nodes.to(device=device, dtype=torch.long)
    adj_mat_edges = adj_mat_edges.to(device=device, dtype=torch.long)

    # Randomly assign each node to a group
    permutation = torch.randperm(num_nodes, device=device)

    group_id = torch.empty(num_nodes, device=device, dtype=torch.long)
    group_id[permutation] = (torch.arange(num_nodes, device=device) // group_size)

    # Retain only edges whose endpoints belong to the same group
    edge_mask = (group_id[adj_mat_nodes] == group_id[adj_mat_edges])

    group_adj_nodes = adj_mat_nodes[edge_mask]
    group_adj_edges = adj_mat_edges[edge_mask]
    group_edge_weights = edge_weights[edge_mask]

    return total_loss(
        batch_x,
        batch_dy_val,
        self_tree,
        inter_tree,
        group_adj_nodes,
        group_adj_edges,
        group_edge_weights,
    )