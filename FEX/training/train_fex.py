from pathlib import Path
from typing import List, Optional

from ..models.learnable_tree import FEX
from .train_configs import FEXConfig, runtimeconfig
import torch
import torch.nn.functional as F
import math
from .loss_funcs import total_loss
from .tree_helpers import copy_fex_state_, get_noise_stds

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
    forcing_tree = forcing_tree.to(device)
    forcing_tree.train()
    inter_dynam_tree = inter_dynam_tree.to(device)
    inter_dynam_tree.train()
    forcing_tree_params = list(forcing_tree.all_parameters())
    inter_tree_params = list(inter_dynam_tree.all_parameters())


    adam_optim_self = torch.optim.Adam(forcing_tree_params, lr=config.lr, betas=(0.95, 0.999))
    adam_optim_inter = torch.optim.Adam(inter_tree_params, lr=config.inter_lr)

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
    adj_matrix = adj_matrix.to(device)
    nodes, edges = adj_matrix.nonzero(as_tuple=True)
    interaction_indices = nodes != edges
    nodes = nodes[interaction_indices].to(device)
    edges = edges[interaction_indices].to(device)

    G = adj_matrix.size(0)
    group_indices = torch.arange(G, device=device)
    scatter_idx = (nodes.unsqueeze(1) == group_indices.unsqueeze(0)).int().argmax(dim=1)

    best_epoch_loss = float('inf')
    if verbose:
        print(f'Initial Equation Forcing Tree: {forcing_tree} \n Inter Tree: {inter_dynam_tree}')

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

            pred_batch_loss = total_loss(batch_x, batch_dy_val, forcing_tree, inter_dynam_tree, nodes, edges, scatter_idx)
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

        if verbose:
            print(f"Adam Epoch {epoch+1}/{config.num_epochs}, PredLoss: {mean_epoch_pred_loss:.4f}")
            print(f'Equation Forcing Tree: {forcing_tree} \n Inter Tree: {inter_dynam_tree}')
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
                # f"self_lr={self_lr:.4g}, inter_lr={inter_lr:.4g}, "
                f"forcing tree: {forcing_tree} \n interaction tree: {inter_dynam_tree}"
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
                pred_error = total_loss(batch_x, batch_dy_val, forcing_tree, inter_dynam_tree, nodes, edges, scatter_idx)
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
                scatter_idx
            ).item()
            for batch_x, batch_dy_val in bfgs_batches 
        ]

        bfgs_loss_val = sum(final_pred_losses) / len(final_pred_losses) 
        if verbose:
            print(
                f"LBFGS refinement: max_iter={config.bfgs_epochs}, loss={bfgs_loss_val:.8e}, "
                f"lr={config.bfgs_lr}, starting_pred_loss={best_epoch_loss:.8e}"
            )
        if bfgs_loss_val < best_epoch_loss:
            best_epoch_loss = bfgs_loss_val

    return float(best_epoch_loss)

def train_fex(forcing_tree, dataloader, config: FEXConfig, device=runtimeconfig.device, verbose=False, every_n_epochs=100):
    forcing_tree.train()
    forcing_tree = forcing_tree.to(device)
    forcing_tree_params = forcing_tree.all_parameters()
    optim = torch.optim.Adam(forcing_tree_params, lr=config.lr)

    # train_logger = runtimeconfig.train_logger if verbose else None

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
            
        if verbose and epoch % every_n_epochs == 0:
            print(f"Epoch {epoch}, Loss: {epoch_loss/max(1, num_batches):.4f}")
            print(f"FEX sequence: {str(forcing_tree.__str__())}")

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
        if verbose:
            print(f"FEX string representation after BFGS optim: {forcing_tree}")
            print(f"Loss after BFGS optim: {bfgs_loss_val}")
        if best_epoch_loss > bfgs_loss_val:
            best_epoch_loss = bfgs_loss_val
    return best_epoch_loss
        


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
