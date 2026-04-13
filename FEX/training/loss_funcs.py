
from typing import List, Optional
from ..models.learnable_tree import FEX
import torch
import torch.nn.functional as F



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
        loss = F.mse_loss(batch_dy, sparse_dx_dt_val)

        # print(f"pred derivative: {batch_dy.mean().item():.4f}, true derivative: {sparse_dx_dt_val.mean().item():.4f}, loss: {loss.item():.4f}")

        return loss



def mag_reverse_l2_regularization(fex: FEX):
    params = fex.tree_mags()
    mags = torch.stack(params)
    reverse_l2 = - (mags ** 2).sum()
    return reverse_l2