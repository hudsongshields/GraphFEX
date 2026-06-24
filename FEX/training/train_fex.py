from pathlib import Path
from typing import List, Optional

from ..models.learnable_tree import FEX
from .train_configs import FEXConfig, runtimeconfig
import torch
import torch.nn.functional as F
import math
from .loss_funcs import group_loss, mag_reverse_l2_regularization, total_loss
from .tree_helpers import copy_fex_state_, get_noise_stds

def leaf_logit_closeness(fex: FEX):
    """Return logit spread (variance around mean) per leaf."""
    total = torch.tensor(0.0, device=next(fex.parameters()).device)
    for leaf in fex.leaf_mlps:
        logits = leaf.logits
        if logits.numel() <= 1:
            continue
        centered = logits - logits.mean()
        total = total + (centered ** 2).mean()
    return total


def train_network_fex(
    forcing_tree: FEX,
    inter_dynam_tree: FEX,
    dataloader,
    adj_matrix,
    config: FEXConfig,
    device=runtimeconfig.device,
    verbose: bool = False,
    log_every: int = 0,
):
    forcing_tree.train()
    inter_dynam_tree.train()
    forcing_tree_params = list(forcing_tree.all_parameters())
    inter_tree_params = list(inter_dynam_tree.all_parameters())


    adam_optim_self = torch.optim.Adam(forcing_tree_params, lr=config.lr, betas=(0.95, 0.999))
    adam_optim_inter = torch.optim.Adam(inter_tree_params, lr=config.inter_lr) #, betas=(0.95, 0.999))

    lr_decay = config.lr_decay
    if lr_decay > 0:
        scheduler_self = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            adam_optim_self,
            T_0=max(int(config.num_epochs * config.pct_cosine_restart), 1),
            eta_min=0.0
        )
        scheduler_inter = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            adam_optim_inter,
            T_0=max(int(config.num_epochs * config.pct_cosine_restart), 1),
            eta_min=0.0
        )


    # Precompute edge indices - used by both group_loss
    nodes, edges = adj_matrix.nonzero(as_tuple=True)
    interaction_indices = nodes != edges
    nodes = nodes[interaction_indices].to(device)
    edges = edges[interaction_indices].to(device)

    G = adj_matrix.size(0)
    group_indices = torch.arange(G, device=device)
    scatter_idx = (nodes.unsqueeze(1) == group_indices.unsqueeze(0)).int().argmax(dim=1)

    best_epoch_loss = float('inf')
    best_forcing_tree = None
    best_inter_tree = None
    train_logger = runtimeconfig.train_logger if verbose else None
    if train_logger:
        train_logger.debug(f'Initial Equation Forcing Tree: {forcing_tree} \n Inter Tree: {inter_dynam_tree}')

    inter_dynam_tree.train()
    forcing_tree.train()
    tau_schedule = getattr(config, "tau_schedule", "exponential")
    tau_max = float(config.tau_start)
    tau_min = float(config.tau_end)
    tau_horizon = max(float(config.tau_anneal_epochs) - 1.0, 1.0)

    tau_decay_rate = None
    if (
        tau_schedule == "exponential"
        and tau_max > 0.0
        and tau_min > 0.0
        and tau_max != tau_min
    ):
        tau_decay_rate = math.log(tau_max / tau_min) / tau_horizon

    for epoch in range(config.num_epochs):
        if tau_schedule == "non_decay":
            current_tau = tau_max
        elif tau_schedule == "linear":
            progress = min(float(epoch) / tau_horizon, 1.0)
            current_tau = tau_max + (tau_min - tau_max) * progress
        elif tau_decay_rate is not None:
            if tau_max > tau_min:
                current_tau = max(tau_min, tau_max * math.exp(-tau_decay_rate * float(epoch)))
            else:
                current_tau = min(tau_min, tau_max * math.exp(-tau_decay_rate * float(epoch)))
        else:
            current_tau = tau_max
        
        if hasattr(forcing_tree, "set_leaf_tau"):
            forcing_tree.set_leaf_tau(current_tau)
        if hasattr(inter_dynam_tree, "set_leaf_tau"):
            inter_dynam_tree.set_leaf_tau(current_tau)

        
        if epoch == int(config.set_hard_at_epoch):
            if hasattr(forcing_tree, "set_leaf_hard"):
                forcing_tree.set_leaf_hard(True)
            if hasattr(inter_dynam_tree, "set_leaf_hard"):
                inter_dynam_tree.set_leaf_hard(True)
        
        forcing_tree.train()
        inter_dynam_tree.train()

        epoch_pred_loss = 0.0
        num_batches = 0
        for batch_x, batch_dy_val in dataloader:
            batch_dy_val = batch_dy_val[:, :, 0:1]  # only learn first dim of dx/dt
            if device == 'cpu':
                batch_x = batch_x.to(device)
                batch_dy_val = batch_dy_val.to(device)
            else:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_dy_val = batch_dy_val.to(device, non_blocking=True)

            adam_optim_self.zero_grad()
            adam_optim_inter.zero_grad()

            # main mse loss
            if config.num_groups == 1:
                pred_batch_loss = total_loss(batch_x, batch_dy_val, forcing_tree, inter_dynam_tree, nodes, edges, scatter_idx)
                batch_loss = pred_batch_loss
                epoch_pred_loss += pred_batch_loss.detach().item()

            # Sign-controlled leaf regularizer:
            use_entropy = config.leaf_entropy_weight != 0
            if use_entropy:
                if config.decay_entropy_until > 0.0:
                    entropy_warmup = config.num_epochs * config.decay_entropy_until
                    entropy_weight = max(0.0, 1.0 - epoch / max(entropy_warmup, 1)) * config.leaf_entropy_weight
                    if entropy_weight != 0:
                        logit_spread = leaf_logit_closeness(forcing_tree) + leaf_logit_closeness(inter_dynam_tree)
                        batch_loss = batch_loss - entropy_weight * logit_spread
                else:
                    logit_spread = leaf_logit_closeness(forcing_tree) + leaf_logit_closeness(inter_dynam_tree)
                    batch_loss = batch_loss - config.leaf_entropy_weight * logit_spread

            # Reverse L2 - encourage non-trivial solutions (inter tree only)
            if (mag_entropy_weight := config.mag_entropy_weight) > 0:
                mag_entropy = mag_reverse_l2_regularization(inter_dynam_tree)
                batch_loss = batch_loss + mag_entropy_weight * mag_entropy

            batch_loss.backward()
            adam_optim_self.step()
            adam_optim_inter.step()

            num_batches += 1
        
        if lr_decay > 0:
            scheduler_self.step()
            scheduler_inter.step()

        mean_epoch_pred_loss = epoch_pred_loss / max(1, num_batches)

        if mean_epoch_pred_loss < best_epoch_loss:
            best_epoch_loss = mean_epoch_pred_loss
            best_forcing_tree = forcing_tree.copy_inorder()
            best_inter_tree = inter_dynam_tree.copy_inorder()

        if train_logger:
            train_logger.debug(f"Epoch {epoch+1}/{config.num_epochs}, Tau: {current_tau:.4f}")
            train_logger.info(f"Adam Epoch {epoch+1}/{config.num_epochs}, PredLoss: {mean_epoch_pred_loss:.4f}")
            train_logger.debug(f'Equation Forcing Tree: {forcing_tree} \n Inter Tree: {inter_dynam_tree}')
        if log_every > 0 and (
            epoch == 0
            or (epoch + 1) % log_every == 0
            or epoch + 1 == config.num_epochs
        ):
            self_lr = adam_optim_self.param_groups[0]["lr"]
            inter_lr = adam_optim_inter.param_groups[0]["lr"]
            print(
                f"Adam epoch {epoch + 1:>5}/{config.num_epochs}: "
                f"pred_loss={mean_epoch_pred_loss:.8e}, "
                f"self_lr={self_lr:.4g}, inter_lr={inter_lr:.4g}, "
                f"tau={current_tau:.4g}"
            )
    
    if config.bfgs_epochs > 0:
        bfgs_forcing_tree = best_forcing_tree.to(device)
        bfgs_inter_tree = best_inter_tree.to(device)
        all_parameters = list(bfgs_forcing_tree.all_parameters()) + list(bfgs_inter_tree.all_parameters())
        bfgs_optim = torch.optim.LBFGS(
            all_parameters,
            lr=config.bfgs_lr,
            max_iter=config.bfgs_epochs,
            line_search_fn="strong_wolfe"
        )

        # Prebuild train set for LBFGS closure
        bfgs_batches = []
        for batch_x, batch_dy_val in dataloader:
            batch_dy_val = batch_dy_val[:, :, 0:1]

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
            for batch_x, batch_dy_val in bfgs_batches:
                pred_error = total_loss(batch_x, batch_dy_val, bfgs_forcing_tree, bfgs_inter_tree, nodes, edges, scatter_idx)
                entropy_error = -config.leaf_entropy_weight * (leaf_logit_closeness(bfgs_forcing_tree) + leaf_logit_closeness(bfgs_inter_tree))
                entropy_error += config.mag_entropy_weight * mag_reverse_l2_regularization(bfgs_inter_tree)
                accumulated_loss = accumulated_loss + pred_error + entropy_error
                total_pred_error = total_pred_error + pred_error.detach().item()

            accumulated_loss.backward()
            return accumulated_loss / len(bfgs_batches)

        if log_every > 0:
            print(
                f"Starting LBFGS refinement: max_iter={config.bfgs_epochs}, "
                f"lr={config.bfgs_lr}, starting_pred_loss={best_epoch_loss:.8e}"
            )
        bfgs_optim.step(closure)
        with torch.no_grad():
            final_pred_losses = [
                total_loss(
                    batch_x,
                    batch_dy_val,
                    bfgs_forcing_tree,
                    bfgs_inter_tree,
                    nodes,
                    edges,
                    scatter_idx,
                ).item()
                for batch_x, batch_dy_val in bfgs_batches
            ]
        bfgs_loss_val = sum(final_pred_losses) / len(final_pred_losses)
        # train_logger.info(f"BFGS Completed (max_iter={config.bfgs_epochs}), Loss: {bfgs_loss_val:.4f}")

        if not math.isfinite(bfgs_loss_val):
            # train_logger.warning(f"BFGS produced non-finite loss: {bfgs_loss_val}")
            if best_epoch_loss != float('inf'):
                # train_logger.info(f"Returning best epoch loss from Adam phase: {best_epoch_loss:.4f}")
                copy_fex_state_(forcing_tree, best_forcing_tree)
                copy_fex_state_(inter_dynam_tree, best_inter_tree)
                return best_epoch_loss
            return float('inf')

        print(f"BFGS PredLoss: {bfgs_loss_val:.4f}, Best Adam PredLoss: {best_epoch_loss:.4f}")
        if bfgs_loss_val < best_epoch_loss:
            best_epoch_loss = bfgs_loss_val
            copy_fex_state_(forcing_tree, bfgs_forcing_tree)
            copy_fex_state_(inter_dynam_tree, bfgs_inter_tree)
        else:
            copy_fex_state_(forcing_tree, best_forcing_tree)
            copy_fex_state_(inter_dynam_tree, best_inter_tree)
    elif best_forcing_tree is not None and best_inter_tree is not None:
        copy_fex_state_(forcing_tree, best_forcing_tree)
        copy_fex_state_(inter_dynam_tree, best_inter_tree)

    return float(best_epoch_loss)


if __name__ == "__main__":
    from torch.utils.data import DataLoader, TensorDataset
    from ..utils.tree_configs import get_tree_config
    import pandas as pd
    import numpy as np
    from ..utils.numerical_deriv import NumericalDeriv
    from .tree_helpers import visualize_tree
    from .debug import debug_tree_configs

    run_str = "debugging/did_i_break_something9"
    save_path = Path(f"FEX/training/testing/{run_str}/final_inter_tree.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    adj_matrix = pd.read_csv('HR/data/BA_Nnodes100_Adj_deg_7_1.csv', header=None)
    num_graph_nodes = adj_matrix.shape[0]

    x_df = pd.read_csv('HR/data/HR_timeseries_SNR_40.csv', header=None)
    num_timesteps, num_cols = x_df.shape
    x_np = x_df.to_numpy(dtype=np.float32)
    x_data = torch.from_numpy(x_np.reshape(num_timesteps, num_graph_nodes, 3))

    dt = 0.01
    len_run = 500
    per_run_timesteps = int(len_run / dt)
    resolution_factor = 1
    dt = dt * resolution_factor

    num_runs = num_timesteps // per_run_timesteps
    x_chunks = torch.chunk(x_data, num_runs, dim=0)
    all_dx_dt = []
    all_x = []

    for x_run in x_chunks:
        x_run = x_run[::resolution_factor]
        dx_dt = NumericalDeriv(x_run, dt=dt) # 4th order
        x_run = x_run[2:-2]
        all_dx_dt.append(dx_dt)
        all_x.append(x_run)
    all_x = all_x[:1] # take the first 1 run only
    all_dx_dt = all_dx_dt[:1] # take the first 1 run only
    dx_dt = torch.cat(all_dx_dt, dim=0)
    x_data = torch.cat(all_x, dim=0)



    train_x_data = x_data[:, :, :]
    train_dx_dt = dx_dt[:, :, :]
    print(f"Training data shape: {train_x_data.shape}, Training dx/dt shape: {train_dx_dt.shape}")
    adj_matrix_tensor = torch.tensor(adj_matrix.values, dtype=torch.float32).to(runtimeconfig.device)
    x_data_tensor_ds = TensorDataset(train_x_data, train_dx_dt)
    if runtimeconfig.device == "cuda":
        dataloader = DataLoader(x_data_tensor_ds, batch_size=512, shuffle=True, pin_memory=True)
    else:
        dataloader = DataLoader(x_data_tensor_ds, batch_size=512, shuffle=True)

    log_path = f"{str(save_path.parent)}/ground_truth_test.log"
    runtimeconfig.CreateLogger(log_path, name="train_logger")

    forcing_tree_config = get_tree_config("depth_3_leaves_4_config")
    inter_tree_config = get_tree_config("depth_2_tree_config")

    fex_config = FEXConfig(
        num_epochs=5000,
        bfgs_epochs=0,
        lr=0.02,
        inter_lr=0.02,
        lr_decay=0.1,

        bfgs_lr=0.1,
        leaf_dim=x_data.shape[2],
        num_leaves=forcing_tree_config.num_leaves,

    )
    fex_config.leaf_entropy_weight = 0.0
    fex_config.decay_entropy_until = 0.0
    fex_config.mag_entropy_weight = 0.0

    fex_config.pct_cosine_restart = 1.0
    


    forcing_op_indices = torch.tensor([0, 0, 0, 0, 1, 2, 0], dtype=torch.long).to(runtimeconfig.device)
    forcing_fex = FEX(
        sample_indices=forcing_op_indices,
        leaf_dim=fex_config.leaf_dim,
        num_leaves=forcing_tree_config.num_leaves,
        tree_structure=forcing_tree_config,
    ).to(runtimeconfig.device)

    inter_op_indices = torch.tensor([1, 0, 3], dtype=torch.long).to(runtimeconfig.device)
    inter_fex = FEX(
        sample_indices=inter_op_indices,
        leaf_dim=fex_config.leaf_dim * 2,
        num_leaves=inter_tree_config.num_leaves,
        tree_structure=inter_tree_config,
    ).to(runtimeconfig.device)

    """ FOR DEBUGGING """
    
    """    forcing_fex = None
    inter_fex = None
    forcing_fex = debug_tree_configs.build_debug_dx_forcing_fex(node_dim=fex_config.leaf_dim, device=runtimeconfig.device)
    inter_fex = debug_tree_configs.build_debug_interaction_fex(node_dim=fex_config.leaf_dim, device=runtimeconfig.device)

    #freeze all tree params
    for param in forcing_fex.all_parameters():
        param.requires_grad = False
    for param in inter_fex.all_parameters():
        param.requires_grad = False
    
    
    # set to random only this param to test if can recover
    with torch.no_grad():
        cubed_term_node = forcing_fex.parent_node.left.right
        # cubed_term_node.operation.a.fill_(torch.empty((1)).uniform_(-2.0, -1.0).item())

        # fill random to see if can recover
        fill_val = torch.randn((1,)).item() * 0.1
        forcing_fex.leaf_mlps[1].logits.fill_(fill_val)


    forcing_fex.leaf_mlps[1].logits.requires_grad = True
    cubed_term_node.operation.a.requires_grad = True"""
    

    loss = 0.0
    loss_history = []
    coeff_history = []
    try:
        loss = train_network_fex(forcing_fex, inter_fex, dataloader, adj_matrix_tensor, fex_config, verbose=True)
    except KeyboardInterrupt:
        print("\nTraining interrupted, saving current state")
        loss = float('inf')
    finally:
        reward = 1/np.sqrt(1 + loss)
        visualize_tree(forcing_fex, f"{str(save_path.parent)}/final_forcing_tree.png")
        visualize_tree(inter_fex, f"{str(save_path.parent)}/final_inter_tree.png")

        from ..utils.pools import GraphPool, GraphPoolCandidate
        pool_candidate = GraphPoolCandidate(
            forcing_tree=forcing_fex.copy_inorder(),
            inter_tree=inter_fex.copy_inorder(),
            reward=reward,
            id=0
        )
        graph_pool = GraphPool(pool_size=1)
        graph_pool.add_new(candidate=pool_candidate)
        graph_pool.save_candidates(directory=f"{str(save_path.parent)}", clear_directory=False)
        print(f"Saved to {save_path.parent}")
