from .nodes import Node
from .learnable_tree import FEX
from typing import Callable
from .train_configs import FEXConfig, RunTimeConfig
import torch
import math
from .train_configs import runtimeconfig

def train_network_fex(forcing_tree: FEX, inter_dynam_tree: FEX, dataloader, adj_matrix, config: FEXConfig, train_logger):
    adam_optim_forcing = torch.optim.Adam(forcing_tree.all_parameters(), lr=config.lr)
    adam_optim_inter = torch.optim.Adam(inter_dynam_tree.all_parameters(), lr=config.lr)
    train_logger.debug(f"starting param forcing tree: {[param.detach().cpu().numpy() for param in forcing_tree.tree_params()]}")
    train_logger.debug(f"starting param inter tree: {[param.detach().cpu().numpy() for param in inter_dynam_tree.tree_params()]}")

    criterion = torch.nn.MSELoss()

    max_nodes = config.max_nodes
    grad_clip_norm = config.max_norm
    best_epoch_loss = float('inf')
    best_forcing_tree = None
    best_inter_tree = None

    for epoch in range(config.num_epochs):
        anneal_epochs = max(int(config.tau_anneal_epochs), 1)
        progress = min(epoch / max(anneal_epochs - 1, 1), 1.0)
        current_tau = config.tau_start + (config.tau_end - config.tau_start) * progress
        forcing_tree.set_leaf_tau(current_tau)
        inter_dynam_tree.set_leaf_tau(current_tau)

        epoch_loss = 0.0
        # batching for SGD
        for batch_x, batch_dy_val in dataloader:
            batch_dy_val = batch_dy_val[:, :, 0] # Only learn first dim of dx/dt
            if runtimeconfig.device == 'cuda':
                batch_x = batch_x.to(runtimeconfig.device, non_blocking=True)
                batch_dy_val = batch_dy_val.to(runtimeconfig.device, non_blocking=True)
            else:
                batch_x = batch_x.to(runtimeconfig.device)
                batch_dy_val = batch_dy_val.to(runtimeconfig.device)

            if batch_x.dim() == 2:
                batch_x = batch_x.unsqueeze(-1) # (batch_size, num_nodes, 1)
            if batch_dy_val.dim() == 2:
                batch_dy_val = batch_dy_val.unsqueeze(-1) # (batch_size, num_nodes, 1)

            # sample subset for stochastic coordinate descent
            random_indices = torch.randperm(batch_x.size(1))[:max_nodes]
            sparse_batch_x = batch_x[:, random_indices, :].to(runtimeconfig.device)
            sparse_adj_matrix = adj_matrix[random_indices][:, random_indices].to(runtimeconfig.device)
            sparse_dx_dt_val = batch_dy_val[:, random_indices, :].to(runtimeconfig.device)

            adam_optim_forcing.zero_grad()
            adam_optim_inter.zero_grad()

            node_outputs = []
            for node in range(sparse_adj_matrix.shape[0]):
                node_input = sparse_batch_x[:, node, :]
                forcing_output = forcing_tree(node_input)
                interaction_output = inter_dynam_tree(node_input)
                interaction_terms = 1

                neighbors = torch.nonzero(sparse_adj_matrix[node], as_tuple=False).flatten()
                for neighbor in neighbors:
                    neighbor = neighbor.item()
                    if neighbor == node:
                        continue
                    neighbor_input = sparse_batch_x[:, neighbor, :]
                    interaction_output = interaction_output + sparse_adj_matrix[node, neighbor] * inter_dynam_tree(neighbor_input)
                    interaction_terms += 1

                interaction_output = interaction_output / interaction_terms

                node_outputs.append(forcing_output + interaction_output)

            batch_dy = torch.stack(node_outputs, dim=1)
            batch_dy = torch.nan_to_num(batch_dy, nan=0.0, posinf=1e6, neginf=-1e6)
            batch_loss = criterion(batch_dy, sparse_dx_dt_val)
                
                    
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(forcing_tree.all_parameters()), max_norm=grad_clip_norm)
            torch.nn.utils.clip_grad_norm_(list(inter_dynam_tree.all_parameters()), max_norm=grad_clip_norm)
            adam_optim_forcing.step()
            adam_optim_inter.step()

            epoch_loss += batch_loss.detach().item()

        if epoch_loss < best_epoch_loss:
            best_epoch_loss = epoch_loss
            best_forcing_tree = forcing_tree.copy_inorder()
            best_inter_tree = inter_dynam_tree.copy_inorder()
        train_logger.debug(f"Adam Epoch {epoch+1}/{config.num_epochs}, Tau: {current_tau:.4f}, Loss: {epoch_loss:.4f}")
    train_logger.debug(f"param forcing tree: {[param.detach().cpu().numpy() for param in forcing_tree.tree_params()]}")
    train_logger.debug(f"param inter tree: {[param.detach().cpu().numpy() for param in inter_dynam_tree.tree_params()]}")

    if not math.isfinite(epoch_loss):
        train_logger.warning(f"Adam Training Completed with non-finite loss: {epoch_loss}")
        if best_epoch_loss != float('inf'):
            train_logger.debug(f"Returning best epoch loss from Adam phase: {best_epoch_loss:.4f}")
            return best_epoch_loss
        return float('inf')
    else:
        train_logger.debug(f"Adam Training Completed, Final Loss: {epoch_loss:.4f}")

    forcing_tree = best_forcing_tree.to(runtimeconfig.device)
    inter_dynam_tree = best_inter_tree.to(runtimeconfig.device)


    all_parameters = list(forcing_tree.all_parameters()) + list(inter_dynam_tree.all_parameters())
    bfgs_optim = torch.optim.LBFGS(
        all_parameters,
        lr=config.bfgs_lr,
        max_iter=config.num_epochs
    )

    # Build a deterministic train set for LBFGS closure
    bfgs_batches = []
    for batch_x, batch_dy_val in dataloader:
        batch_dy_val = batch_dy_val[:, :, 0] # Only learn first dim of dx/dt
        if runtimeconfig.device == 'cuda':
            batch_x = batch_x.to(runtimeconfig.device, non_blocking=True)
            batch_dy_val = batch_dy_val.to(runtimeconfig.device, non_blocking=True)
        else:
            batch_x = batch_x.to(runtimeconfig.device)
            batch_dy_val = batch_dy_val.to(runtimeconfig.device)

        if batch_x.dim() == 2:
            batch_x = batch_x.unsqueeze(-1) # (batch_size, num_nodes, 1)
        if batch_dy_val.dim() == 2:
            batch_dy_val = batch_dy_val.unsqueeze(-1) # (batch_size, num_nodes, 1)

        fixed_indices = torch.arange(min(max_nodes, batch_x.size(1)), device=runtimeconfig.device)
        sparse_batch_x = batch_x[:, fixed_indices, :]
        sparse_adj_matrix = adj_matrix[fixed_indices][:, fixed_indices].to(runtimeconfig.device)
        sparse_dx_dt_val = batch_dy_val[:, fixed_indices, :]

        bfgs_batches.append((sparse_batch_x, sparse_adj_matrix, sparse_dx_dt_val))

    def closure():
        bfgs_optim.zero_grad()

        total_loss = []
        for sparse_batch_x, sparse_adj_matrix, sparse_dx_dt_val in bfgs_batches:

            node_outputs = []
            for node in range(sparse_adj_matrix.size(0)):
                node_input = sparse_batch_x[:, node, :]
                forcing_output = forcing_tree(node_input)
                interaction_output = inter_dynam_tree(node_input)
                interaction_terms = 1

                neighbors = torch.nonzero(sparse_adj_matrix[node], as_tuple=False).flatten()
                for neighbor in neighbors:
                    neighbor = neighbor.item()
                    if neighbor == node:
                        continue
                    neighbor_input = sparse_batch_x[:, neighbor, :]
                    interaction_output = interaction_output + sparse_adj_matrix[node, neighbor] * inter_dynam_tree(neighbor_input)
                    interaction_terms += 1

                interaction_output = interaction_output / interaction_terms

                node_outputs.append(forcing_output + interaction_output)

            batch_dy = torch.stack(node_outputs, dim=1)
            batch_dy = torch.nan_to_num(batch_dy, nan=0.0, posinf=1e6, neginf=-1e6)

            batch_loss = criterion(batch_dy, sparse_dx_dt_val)
            total_loss.append(batch_loss)

        total_loss = torch.stack(total_loss).mean()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(all_parameters, max_norm=grad_clip_norm)
        return total_loss

    bfgs_loss = bfgs_optim.step(closure)
    bfgs_loss_val = bfgs_loss.item() if isinstance(bfgs_loss, torch.Tensor) else float(bfgs_loss)
    train_logger.debug(f"BFGS Completed (max_iter={config.num_epochs}), Loss: {bfgs_loss_val:.4f}")
    if not math.isfinite(bfgs_loss_val):
        train_logger.warning(f"BFGS produced non-finite loss: {bfgs_loss_val}")
        if best_epoch_loss != float('inf'):
            train_logger.info(f"Returning best epoch loss from Adam phase: {best_epoch_loss:.4f}")
            return best_epoch_loss
        return float('inf')
    if bfgs_loss_val < best_epoch_loss:
        best_epoch_loss = bfgs_loss_val
        best_forcing_tree = forcing_tree.copy_inorder()
        best_inter_tree = inter_dynam_tree.copy_inorder()

    forcing_tree = best_forcing_tree.to(runtimeconfig.device)
    inter_dynam_tree = best_inter_tree.to(runtimeconfig.device)
    return float(best_epoch_loss)

