from pathlib import Path
from typing import List, Optional

from ..models.learnable_tree import FEX
from .train_configs import FEXConfig
import torch
import torch.nn.functional as F
import math
from .loss_funcs import group_loss, total_loss
from .tree_helpers import copy_fex_state_, get_noise_stds


def _expression_summary(tree: FEX):
    if hasattr(tree, "expression_summary"):
        return tree.expression_summary()
    return str(tree)


def train_network_fex(
    forcing_tree: FEX,
    inter_dynam_tree: FEX,
    dataloader,
    adj_matrix,
    config: FEXConfig,
    *,
    num_groups=1,
    device="cuda" if torch.cuda.is_available() else "cpu",
    verbose: bool = False,
    log_every: int = 0,
):
    forcing_tree = forcing_tree.to(device)
    forcing_tree.train()
    inter_dynam_tree = inter_dynam_tree.to(device)
    inter_dynam_tree.train()
    forcing_tree_params = list(forcing_tree.all_parameters())
    inter_tree_params = list(inter_dynam_tree.all_parameters())


    adam_optim_self = torch.optim.Adam(forcing_tree_params, lr=config.lr)
    adam_optim_inter = torch.optim.Adam(inter_tree_params, lr=config.inter_lr)


    # Precompute edge indices - used by both group_loss
    adj_matrix = adj_matrix.to(device)
    nodes, edges = adj_matrix.nonzero(as_tuple=True)
    interaction_indices = nodes != edges
    nodes = nodes[interaction_indices].to(device)
    edges = edges[interaction_indices].to(device)
    edge_weights = adj_matrix[nodes, edges] # nonzero Aij values

    best_epoch_loss = float('inf')

    inter_dynam_tree.train()
    forcing_tree.train()
    for epoch in range(config.num_epochs):
        epoch_pred_loss = 0.0
        num_batches = 0
        for batch_x, batch_dy_val in dataloader:
            batch_dy_val = batch_dy_val[:, :, config.target_dim:config.target_dim+1]
            if device == 'cuda':
                batch_x = batch_x.to(device, non_blocking=True)
                batch_dy_val = batch_dy_val.to(device, non_blocking=True)
            else:
                batch_x = batch_x.to(device)
                batch_dy_val = batch_dy_val.to(device)

            adam_optim_self.zero_grad()
            adam_optim_inter.zero_grad()

            if num_groups > 1:
                pred_batch_loss = group_loss(batch_x, batch_dy_val, forcing_tree, inter_dynam_tree, nodes, edges, edge_weights, num_groups)
            else:
                pred_batch_loss = total_loss(batch_x, batch_dy_val, forcing_tree, inter_dynam_tree, nodes, edges, edge_weights)
            if not torch.isfinite(pred_batch_loss):
                continue
            batch_loss = pred_batch_loss
            epoch_pred_loss += pred_batch_loss.detach().item()

            batch_loss.backward()
            adam_optim_self.step()
            adam_optim_inter.step()

            num_batches += 1
            
        if num_batches == 0:
            continue
        mean_epoch_pred_loss = epoch_pred_loss / max(1, num_batches)

        if mean_epoch_pred_loss < best_epoch_loss:
            best_epoch_loss = mean_epoch_pred_loss


        if log_every > 0 and (
            (epoch + 1) % log_every == 0
            or epoch + 1 == config.num_epochs
        ):

            print(
                f"Adam epoch {epoch + 1:>5}/{config.num_epochs}: "
                f"pred_loss={mean_epoch_pred_loss:.8e}, "
            )
    
    if config.bfgs_epochs > 0:
        all_parameters = list(forcing_tree.all_parameters()) + list(inter_dynam_tree.all_parameters())
        bfgs_optim = torch.optim.LBFGS(
            all_parameters,
            lr=config.bfgs_lr,
            max_iter=config.bfgs_epochs,
        )

        # Prebuild train set for LBFGS closure
        bfgs_batches = []
        for batch_x, batch_dy_val in dataloader:
            batch_dy_val = batch_dy_val[:, :, config.target_dim:config.target_dim+1]

            if device == 'cuda':
                batch_x = batch_x.to(device, non_blocking=True)
                batch_dy_val = batch_dy_val.to(device, non_blocking=True)
            else:
                batch_x = batch_x.to(device)
                batch_dy_val = batch_dy_val.to(device)

            bfgs_batches.append((batch_x, batch_dy_val))

        def closure():
            bfgs_optim.zero_grad()

            accumulated_loss = 0.0
            valid_batches = 0
            for batch_x, batch_dy_val in bfgs_batches:
                pred_error = total_loss(batch_x, batch_dy_val, forcing_tree, inter_dynam_tree, nodes, edges, edge_weights)
                if not torch.isfinite(pred_error):
                    continue
                valid_batches += 1
                accumulated_loss = accumulated_loss + pred_error

            if valid_batches == 0:
                return torch.tensor(float('inf'), device=device)
            accumulated_loss.backward()
            return accumulated_loss / valid_batches

        bfgs_optim.step(closure)

        forcing_tree.eval()
        inter_dynam_tree.eval()
        final_pred_losses = [
            total_loss(
                batch_x,
                batch_dy_val,
                forcing_tree,
                inter_dynam_tree,
                nodes,
                edges,
                edge_weights,
            ).item()
            for batch_x, batch_dy_val in bfgs_batches 
        ]

        bfgs_loss_val = sum(final_pred_losses) / len(final_pred_losses) 
        if bfgs_loss_val < best_epoch_loss:
            best_epoch_loss = bfgs_loss_val
    
    if log_every > 0:
        print(
            f"loss={best_epoch_loss:.8e}"
            f"FEX sequence: {_expression_summary(forcing_tree)} FEX operator sequence: {forcing_tree.sample_indices}\n"
            f"Inter FEX sequence: {_expression_summary(inter_dynam_tree)} Inter FEX operator sequence: {inter_dynam_tree.sample_indices}\n"
        )

    return float(best_epoch_loss)

def train_fex(forcing_tree, dataloader, config: FEXConfig, device="cuda" if torch.cuda.is_available() else "cpu", verbose=False, every_n_epochs=0):
    forcing_tree.train()
    forcing_tree = forcing_tree.to(device)
    forcing_tree_params = forcing_tree.all_parameters()
    optim = torch.optim.Adam(forcing_tree_params, lr=config.lr)


    best_epoch_loss = float('inf')
    for epoch in range(config.num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        for batch_x, batch_dy_val in dataloader:
            batch_x = batch_x.to(device)
            batch_dy_val = batch_dy_val[:, :, config.target_dim:config.target_dim+1].to(device)

            optim.zero_grad()
            loss = total_loss(batch_x, batch_dy_val, forcing_tree)
            if not loss.requires_grad:
                return float('inf')
            if not torch.isfinite(loss):
                return float('inf')
            loss.backward()
            optim.step()

            epoch_loss += loss.item()
            num_batches += 1
        
        if epoch_loss / num_batches < best_epoch_loss:
            best_epoch_loss = epoch_loss / max(1, num_batches)
            
        if every_n_epochs and (epoch+1) % every_n_epochs == 0:
            print(f"Epoch {epoch+1}, Loss: {epoch_loss/max(1, num_batches):.4f}")

    if config.bfgs_epochs > 0:
        all_parameters = list(forcing_tree.all_parameters())
        bfgs_optim = torch.optim.LBFGS(
            all_parameters,
            lr=config.bfgs_lr,
            max_iter=config.bfgs_epochs,
            line_search_fn="strong_wolfe"
        )

        bfgs_batches = []
        for batch_x, batch_dy_val in dataloader:
            batch_dy_val = batch_dy_val[:, :, config.target_dim:config.target_dim+1]

            if device == 'cuda':
                batch_x = batch_x.to(device, non_blocking=True)
                batch_dy_val = batch_dy_val.to(device, non_blocking=True)
            else:
                batch_x = batch_x.to(device)
                batch_dy_val = batch_dy_val.to(device)

            bfgs_batches.append((batch_x, batch_dy_val))

        def closure():
            bfgs_optim.zero_grad()

            accumulated_loss = 0.0
            total_pred_error = 0.0
            valid_batches = 0
            for batch_x, batch_dy_val in bfgs_batches:
                pred_error = total_loss(batch_x, batch_dy_val, forcing_tree)
                if not torch.isfinite(pred_error):
                    continue
                accumulated_loss = accumulated_loss + pred_error
                valid_batches += 1

            if valid_batches == 0:
                return torch.tensor(float('inf'), device=device)
            accumulated_loss.backward()
            
            return accumulated_loss / max(1, valid_batches)

        bfgs_optim.step(closure)

        with torch.no_grad():
            final_pred_losses = [
                total_loss(
                    batch_x,
                    batch_dy_val,
                    forcing_tree,
                ).item()
                for batch_x, batch_dy_val in bfgs_batches
            ]
        bfgs_loss_val = sum(final_pred_losses) / len(final_pred_losses) 
        if best_epoch_loss > bfgs_loss_val:
            best_epoch_loss = bfgs_loss_val

    if every_n_epochs > 0:
        print(f"Final FEX: {_expression_summary(forcing_tree)}")
        print(f"Final Loss: {best_epoch_loss}")
        print(f"fex operator sequence: {forcing_tree.sample_indices}")
    return best_epoch_loss
        
