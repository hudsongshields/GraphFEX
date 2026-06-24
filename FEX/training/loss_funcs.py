
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

def total_loss(batch_x, batch_dy_val, self_tree, inter_tree, adj_mat_nodes, adj_mat_edges, scatter_idx=None):

        B = batch_x.size(0)
        G = batch_x.size(1)
        group_indices = torch.arange(G, device=batch_x.device)
        num_edges = adj_mat_edges.size(0)

        # Batched self interactions
        self_tree = self_tree.to(batch_x.device)
        inter_tree = inter_tree.to(batch_x.device)
        group_inputs = batch_x[:, group_indices, :].reshape(B * G, -1).to(batch_x.device) # input nodes across batch and group of nodes
        forcing_out = self_tree(group_inputs)
        forcing_out = forcing_out.reshape(B, G, 1)
        
        # combine node/edge features 
        inter_sources = batch_x[:, adj_mat_nodes, :]
        inter_edges = batch_x[:, adj_mat_edges, :]
        edge_inputs = torch.cat([inter_sources, inter_edges], dim=-1)
        inter_out = inter_tree(edge_inputs.reshape(B * num_edges, -1))
        inter_out = inter_out.reshape(B, num_edges, 1)

        # scatter-sum interaction outputs to group positions
        if scatter_idx is None:
            scatter_idx = (adj_mat_nodes.unsqueeze(1) == group_indices.unsqueeze(0)).int().argmax(dim=1)
        local_idx = scatter_idx.unsqueeze(0).unsqueeze(-1).expand(B, num_edges, 1)
        interaction_out = torch.zeros(B, G, 1, device=batch_x.device)
        interaction_out.scatter_add_(1, local_idx, inter_out) # sum contributions from all edges to each node

        batch_dy = forcing_out + interaction_out
        loss = F.mse_loss(batch_dy, batch_dy_val[:, :, :])

        # inter_loss = F.mse_loss(interaction_out, (batch_dy_val[:, :, :] - forcing_out))

        if not torch.isfinite(batch_dy).all():
                loss = torch.tensor(float('inf'), device=batch_x.device)

        return loss # , inter_loss





def group_loss(batch_x, batch_dy_val, group_indices, self_tree, inter_tree, adj_mat_nodes, adj_mat_edges):

        sparse_dx_dt_val = batch_dy_val[:, group_indices, :]

        B = batch_x.size(0)
        G = group_indices.size(0)

        # Batched self interactions
        group_inputs = batch_x[:, group_indices, :].reshape(B * G, -1) # input nodes across batch and group of nodes
        forcing_out = self_tree(group_inputs)
        forcing_out = forcing_out.reshape(B, G, 1)

        # Batched interaction - only consider edges with node indices in the selected group (doesnt apply if not using SCD)
        aligned_indices = (adj_mat_nodes.unsqueeze(1) == group_indices.unsqueeze(0)).any(dim=1) 
        sparse_nodes = adj_mat_nodes[aligned_indices]
        sparse_edges = adj_mat_edges[aligned_indices]
        num_sparse_edges = sparse_edges.size(0)
        

        # Gather edge features and forward in one call
        inter_sources = batch_x[:, sparse_nodes, :]
        inter_edges = batch_x[:, sparse_edges, :]
        edge_inputs = torch.cat([inter_sources, inter_edges], dim=-1)
        inter_out = inter_tree(edge_inputs.reshape(B * num_sparse_edges, -1))
        inter_out = inter_out.reshape(B, num_sparse_edges, 1)

        # scatter-sum interaction outputs to group positions
        sparse_indices = (sparse_nodes.unsqueeze(1) == group_indices.unsqueeze(0)).int().argmax(dim=1)
        interaction_out = torch.zeros(B, G, 1, device=batch_x.device)
        local_idx = sparse_indices.unsqueeze(0).unsqueeze(-1).expand(B, num_sparse_edges, 1)
        interaction_out.scatter_add_(1, local_idx, inter_out) # sum contributions from all edges to each node

        batch_dy = forcing_out + interaction_out
        loss = F.mse_loss(batch_dy, sparse_dx_dt_val)

        return loss



def mag_reverse_l2_regularization(fex: FEX):
    params = fex.tree_mags()
    mags = torch.stack(params)
    reverse_l2 = - (mags ** 2).sum()
    return reverse_l2