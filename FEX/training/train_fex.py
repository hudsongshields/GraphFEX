from pathlib import Path
from typing import List, Optional

from ..models.learnable_tree import FEX
from .train_configs import FEXConfig, runtimeconfig
import torch
import torch.nn.functional as F
import math
from .loss_funcs import euler_loss, group_loss, mag_reverse_l2_regularization
from .tree_helpers import get_noise_stds

def leaf_entropy(fex: FEX):
    """Sum of entropies over all leaf logit distributions """
    total = torch.tensor(0.0, device=next(fex.parameters()).device)
    for leaf in fex.leaf_mlps:
        probs = F.softmax(leaf.logits, dim=-1)
        total = total - (probs * (probs + 1e-8).log()).sum()
    return total


def train_network_fex(forcing_tree: FEX, inter_dynam_tree: FEX, dataloader, adj_matrix, config: FEXConfig, x_sequential: Optional[List[torch.Tensor]] = None, use_entropy: bool = True):
    forcing_tree.train()
    inter_dynam_tree.train()
    forcing_tree_params = list(forcing_tree.all_parameters())
    inter_tree_params = list(inter_dynam_tree.all_parameters())

    adam_optim_self = torch.optim.Adam(forcing_tree_params, lr=config.lr, betas=(0.2, 0.999), weight_decay=0)
    adam_optim_inter = torch.optim.Adam(inter_tree_params, lr=config.lr, betas=(0.2, 0.999), weight_decay=0)
    train_logger = runtimeconfig.train_logger

    scheduler_self = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        adam_optim_self,
        T_0=max(int(config.num_epochs * config.pct_cosine_restart), 1),
        eta_min=config.lr * 0.15
    )
    scheduler_inter = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        adam_optim_inter,
        T_0=max(int(config.num_epochs * config.pct_cosine_restart), 1),
        eta_min=config.lr * 0.15
    )

    # For plotting
    loss_history = []
    coeff_history = []


    # Precompute edge indices (no self-loops) - used by both group_loss and euler_loss
    nodes, edges = adj_matrix.nonzero(as_tuple=True)
    interaction_indices = nodes != edges
    nodes = nodes[interaction_indices].to(runtimeconfig.device)
    edges = edges[interaction_indices].to(runtimeconfig.device)

    use_rollout = x_sequential is not None and config.rollout_weight > 0


    num_groups = config.num_groups
    best_epoch_loss = float('inf')
    best_forcing_tree = None
    best_inter_tree = None

    train_logger.debug(f'Initial Equation Forcing Tree: {forcing_tree} \n Inter Tree: {inter_dynam_tree}')


    inter_dynam_tree.train()
    forcing_tree.train()
    for epoch in range(config.num_epochs):
        anneal_epochs = config.tau_anneal_epochs
        progress = min(epoch / max(anneal_epochs - 1, 1), 1.0)
        current_tau = config.tau_start + (config.tau_end - config.tau_start) * progress
        forcing_tree.set_leaf_tau(current_tau)
        inter_dynam_tree.set_leaf_tau(current_tau)
        train_logger.debug(f"Epoch {epoch+1}/{config.num_epochs}, Tau: {current_tau:.4f}")

        if epoch ==int(config.num_epochs * config.pct_cosine_restart * 0.0):
            """forcing_tree.unfreeze_b()
            inter_dynam_tree.unfreeze_b()"""
        
        if epoch == int(config.num_epochs * config.set_hard_at):
            forcing_tree.set_leaf_hard(True)
            inter_dynam_tree.set_leaf_hard(True)
        
        forcing_tree.train()
        inter_dynam_tree.train()

        epoch_loss = 0.0
        num_batches = 0
        for batch_x, batch_dy_val in dataloader:
            batch_dy_val = batch_dy_val[:, :, 0:1]  # only learn first dim of dx/dt

            if runtimeconfig.device == 'cuda':
                batch_x = batch_x.to(runtimeconfig.device, non_blocking=True)
                batch_dy_val = batch_dy_val.to(runtimeconfig.device, non_blocking=True)
            else:
                batch_x = batch_x.to(runtimeconfig.device)
                batch_dy_val = batch_dy_val.to(runtimeconfig.device)

            group_indices = torch.arange(batch_x.size(1), device=batch_x.device)  # use all nodes


            adam_optim_self.zero_grad()
            adam_optim_inter.zero_grad()

            batch_loss = group_loss(batch_x, batch_dy_val, group_indices, forcing_tree, inter_dynam_tree, nodes, edges)

            # Entropy - encourage dim exploration early, decay to 0
            if use_entropy:
                entropy_warmup = config.tau_anneal_epochs
                entropy_weight = max(0.0, 1.0 - epoch / max(entropy_warmup, 1)) * 0.65
                if entropy_weight > 0:
                    entropy = leaf_entropy(forcing_tree) + leaf_entropy(inter_dynam_tree)
                    batch_loss = batch_loss - entropy_weight * entropy

            # Euler loss – Gaussian bell curve centered at epoch 500, skip early epochs entirely
            if use_rollout:
                center = config.num_epochs * 0.5
                sigma = config.num_epochs / 6.0
                if epoch >= center - 2.5 * sigma and epoch <= center + 2.5 * sigma:
                    rollout_weight = config.rollout_weight * math.exp(-0.5 * ((epoch - center) / sigma) ** 2)
                    r_loss = euler_loss(
                        x_sequential, config.rollout_dt,
                        forcing_tree, inter_dynam_tree, adj_matrix,
                        nodes, edges,
                        rollout_steps=config.rollout_steps,
                    )
                    # Log-scale: compress large Euler losses while preserving relative ordering
                    r_loss = torch.log1p(r_loss)
                    batch_loss = batch_loss + r_loss * rollout_weight

            if (mag_entropy_weight := config.mag_entropy_weight) > 0:
                mag_entropy = 0.0
                mag_entropy += mag_reverse_l2_regularization(forcing_tree)
                mag_entropy += mag_reverse_l2_regularization(inter_dynam_tree)
                batch_loss = batch_loss + mag_entropy_weight * mag_entropy

            batch_loss.backward()
            adam_optim_self.step()
            adam_optim_inter.step()

            epoch_loss += batch_loss.detach().item()
            num_batches += 1


        # DEBUG: Print gradients for key parameters (e.g., cubed_term_node.operation.log_mag)
        cubed_term_node = forcing_tree.parent_node.left.right
        log_mag_grad = cubed_term_node.operation.log_mag.grad
        sign_logit_grad = cubed_term_node.operation.sign_logit.grad
        leaf_logits_grad = forcing_tree.leaf_mlps[1].logits.grad
        print(f"[DEBUG] log_mag grad: {log_mag_grad}")
        print(f"[DEBUG] sign_logit grad: {sign_logit_grad}")
        print(f"[DEBUG] leaf logits grad: {leaf_logits_grad}")

        
        scheduler_self.step()
        scheduler_inter.step()


        mean_epoch_loss = epoch_loss / max(1, num_batches)

        # Track loss and coefficient
        loss_history.append(mean_epoch_loss)
        # Try to get the coefficient (log_mag) for plotting
        try:
            cubed_term_node = forcing_tree.parent_node.left.right
            coeff_history.append(float(cubed_term_node.operation.log_mag.detach().cpu().item()))
        except Exception:
            coeff_history.append(float('nan'))

        if mean_epoch_loss < best_epoch_loss:
            best_epoch_loss = mean_epoch_loss
            best_forcing_tree = forcing_tree.copy_inorder()
            best_inter_tree = inter_dynam_tree.copy_inorder()

        train_logger.info(f"Adam Epoch {epoch+1}/{config.num_epochs}, Loss: {mean_epoch_loss:.4f}")
        train_logger.debug(f"Current equation Forcing Tree: {forcing_tree} \n Inter Tree: {inter_dynam_tree}")
        # train_logger.debug(f"Noise std Forcing Tree: {', '.join(f'{leaf}: {noise.item():.4f}' for leaf, noise in get_noise_stds(forcing_tree).items())} \n Inter Tree: {', '.join(f'{leaf}: {noise.item():.4f}' for leaf, noise in get_noise_stds(inter_dynam_tree).items())}")

    return float(best_epoch_loss), loss_history, coeff_history

    if not math.isfinite(epoch_loss):
        train_logger.warning(f"Adam Training Completed with non-finite loss: {epoch_loss}")
        if best_epoch_loss != float('inf'):
            train_logger.debug(f"Returning best epoch loss from Adam phase: {best_epoch_loss:.4f}")
            return best_epoch_loss
        return float('inf')
    else:
        train_logger.debug(f"Adam Training Completed, Final Loss: {best_epoch_loss:.4f}")

    forcing_tree = best_forcing_tree.to(runtimeconfig.device)
    inter_dynam_tree = best_inter_tree.to(runtimeconfig.device)

    if config.bfgs_epochs > 0:
        all_parameters = list(forcing_tree.all_parameters()) + list(inter_dynam_tree.all_parameters())
        bfgs_optim = torch.optim.LBFGS(
            all_parameters,
            lr=config.bfgs_lr,
            max_iter=config.bfgs_epochs
        )

        # Prebuild train set for LBFGS closure
        bfgs_batches = []
        for batch_x, batch_dy_val in dataloader:
            batch_dy_val = batch_dy_val[:, :, 0:1]

            if runtimeconfig.device == 'cuda':
                batch_x = batch_x.to(runtimeconfig.device, non_blocking=True)
                batch_dy_val = batch_dy_val.to(runtimeconfig.device, non_blocking=True)
            else:
                batch_x = batch_x.to(runtimeconfig.device)
                batch_dy_val = batch_dy_val.to(runtimeconfig.device)

            """random_indices = torch.randperm(batch_x.size(1), device=batch_x.device)
            group_indices = torch.chunk(random_indices, num_groups)
            random_group = torch.randint(0, num_groups, (1,), device=batch_x.device).item()
            group_indices = group_indices[random_group]"""
            group_indices = torch.arange(batch_x.size(1), device=batch_x.device)  # use all nodes for LBFGS

            bfgs_batches.append((batch_x, batch_dy_val, group_indices))

        def closure():
            bfgs_optim.zero_grad()

            total_loss = 0.0
            for batch_x, batch_dy_val, group_indices in bfgs_batches:
                batch_loss = group_loss(batch_x, batch_dy_val, group_indices, forcing_tree, inter_dynam_tree, nodes, edges)
                total_loss = total_loss + batch_loss

            total_loss.backward()
            return total_loss

        bfgs_loss = bfgs_optim.step(closure)
        bfgs_loss_val = bfgs_loss.item() if isinstance(bfgs_loss, torch.Tensor) else float(bfgs_loss)
        train_logger.info(f"BFGS Completed (max_iter={config.bfgs_epochs}), Loss: {bfgs_loss_val:.4f}")

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
    else:
        train_logger.info("Skipping BFGS phase as bfgs_epochs is set to 0")

    forcing_tree = best_forcing_tree.to(runtimeconfig.device)
    inter_dynam_tree = best_inter_tree.to(runtimeconfig.device)

    train_logger.debug(f"Final Forcing Tree Equation: {forcing_tree}")
    train_logger.debug(f"Final Inter Tree Equation: {inter_dynam_tree}")

    return float(best_epoch_loss)


if __name__ == "__main__":
    from torch.utils.data import DataLoader, TensorDataset
    from ..utils.tree_configs import get_tree_config
    import pandas as pd
    import numpy as np
    from ..utils.numerical_deriv import NumericalDeriv
    from .tree_helpers import visualize_tree
    from .debug import debug_tree_configs

    run_str = "debugging/recover_cubed_coeff_and_dim"
    save_path = Path(f"FEX/training/testing/{run_str}/final_inter_tree.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    adj_matrix = pd.read_csv('HR/data/BA_Nnodes100_Adj_deg_7_1.csv', header=None)
    num_graph_nodes = adj_matrix.shape[0]

    x_df = pd.read_csv('HR/new_data/HR_timeseries_BA_deg_7_1_runs_5.csv', header=None)
    num_timesteps, num_cols = x_df.shape
    x_np = x_df.to_numpy(dtype=np.float32)
    x_data = torch.from_numpy(x_np.reshape(num_timesteps, num_graph_nodes, 3))

    dt = 0.01
    len_run = 100
    per_run_timesteps = int(len_run / dt)
    cut_timestep = per_run_timesteps * 1.0
    
    num_runs = num_timesteps // per_run_timesteps
    x_chunks = torch.chunk(x_data, num_runs, dim=0)
    all_dx_dt = []
    all_x = []

    for x_run in x_chunks:
        x_run = x_run[:int(cut_timestep)]
        dx_dt = NumericalDeriv(x_run, dt=dt) # 4th order
        x_run = x_run[2:-2]
        all_dx_dt.append(dx_dt)
        all_x.append(x_run)
    all_x = all_x[:1]   # take the first 1 run only
    all_dx_dt = all_dx_dt[:1]   # take the first 1 run only
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
        num_epochs = 1000,
        bfgs_epochs = 0,
        lr = 0.02,
        leaf_lr = 0.02,
        bfgs_lr = 0.1,
        num_groups = 1,
        leaf_dim = x_data.shape[2],
        num_leaves = forcing_tree_config.num_leaves
    )
    fex_config.tau_anneal_epochs = fex_config.num_epochs * 0.75
    fex_config.tau_start = 8.0
    fex_config.pct_cosine_restart = 0.25
    fex_config.set_hard_at = 0.5

    fex_config.mag_entropy_weight = 0.9

    forcing_op_indices = torch.tensor([0, 0, 0, 0, 2, 1, 0], dtype=torch.long).to(runtimeconfig.device)
    forcing_fex = FEX(
        sample_indices=forcing_op_indices,
        leaf_dim=fex_config.leaf_dim,
        num_leaves=forcing_tree_config.num_leaves,
        tree_structure=forcing_tree_config,
        init_tau=fex_config.tau_start,
    ).to(runtimeconfig.device)

    inter_op_indices = torch.tensor([1, 0, 5], dtype=torch.long).to(runtimeconfig.device)
    inter_fex = FEX(
        sample_indices=inter_op_indices,
        leaf_dim=fex_config.leaf_dim * 2,
        num_leaves=inter_tree_config.num_leaves,
        tree_structure=inter_tree_config,
        init_tau=fex_config.tau_start
    ).to(runtimeconfig.device)


    """ FOR DEBUGGING """
    
    forcing_fex = None
    inter_fex = None
    forcing_fex = debug_tree_configs.build_debug_dx_forcing_fex(node_dim=fex_config.leaf_dim, device=runtimeconfig.device)
    inter_fex = debug_tree_configs.build_debug_interaction_fex(node_dim=fex_config.leaf_dim, device=runtimeconfig.device)

    #freeze all tree params
    for param in forcing_fex.all_parameters():
        param.requires_grad = False
    for param in inter_fex.all_parameters():
        param.requires_grad = False
    
    """    
    forcing_fex.set_leaf_hard(True)
    inter_fex.set_leaf_hard(True)"""
    
    # set to random only this param to test if can recover
    with torch.no_grad():
        log_mag = torch.empty((1)).uniform_(0.1, 1.0).log()
        sign_logit = torch.ones(1) * (0.1)
        cubed_term_node = forcing_fex.parent_node.left.right
        cubed_term_node.operation.log_mag.fill_(log_mag.item())
        cubed_term_node.operation.sign_logit.fill_(sign_logit.item())

        # fill random to see if can recover
        fill_val = torch.randn((1,)).item() * 0.1
        forcing_fex.leaf_mlps[1].logits.fill_(fill_val)

        """for leaf in forcing_fex.leaf_mlps:
            leaf.sigma.fill_(-100.0)"""

    forcing_fex.leaf_mlps[1].logits.requires_grad = True
    cubed_term_node.operation.log_mag.requires_grad = True
    cubed_term_node.operation.sign_logit.requires_grad = True
    

    loss = 0.0
    loss_history = []
    coeff_history = []
    try:
        loss, loss_history, coeff_history = train_network_fex(forcing_fex, inter_fex, dataloader, adj_matrix_tensor, fex_config, x_sequential=None, use_entropy=False)
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

        # Plot loss, coefficient, true and predicted dy/dx
        import matplotlib.pyplot as plt
        # True dy/dx (dim 1)
        true_dydx = dx_dt[:, :, 1].reshape(-1).cpu().numpy()
        # Predicted dy/dx (dim 1)

        with torch.no_grad():
            pred_dydx = []
            device = next(forcing_fex.parameters()).device
            for i in range(x_data.shape[0]):
                pred = forcing_fex(x_data[i].to(device))
                if pred.ndim == 2:
                    # If only one feature, use column 0; else, use column 1
                    if pred.shape[1] == 1:
                        pred_dim1 = pred[:, 0].cpu().numpy()
                    elif pred.shape[1] > 1:
                        pred_dim1 = pred[:, 1].cpu().numpy()
                    else:
                        raise ValueError(f"Unexpected pred shape: {pred.shape}")
                elif pred.ndim == 1:
                    # fallback: single node
                    if pred.shape[0] > 1:
                        pred_dim1 = np.array([pred[1].cpu().item()])
                    else:
                        pred_dim1 = np.array([pred[0].cpu().item()])
                else:
                    raise ValueError(f"Unexpected pred shape: {pred.shape}")
                pred_dydx.append(pred_dim1)
            pred_dydx = np.concatenate(pred_dydx, axis=0)  # shape: (num_timesteps * num_nodes,)

        import torch.nn.functional as F
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes[0, 0].plot(loss_history, label='Loss')
        axes[0, 0].set_title('Loss Curve')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()

        axes[0, 1].plot(F.softplus(torch.tensor(coeff_history)), label='Coefficient (log_mag)')
        axes[0, 1].set_title('Coefficient Value')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Value')
        axes[0, 1].legend()

        axes[1, 0].plot(true_dydx[:], label='True dy/dx (dim 1)')
        axes[1, 0].plot(pred_dydx[:], label='Predicted dy/dx (dim 1)', linestyle='dashed', color='orange', alpha=0.7, linewidth=2)
        axes[1, 0].plot(true_dydx[:], label='True dy/dx (dim 1)', color='blue', linewidth=3, zorder=10)
        axes[1, 0].set_title('True vs Predicted dy/dx (dim 1)')
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('dy/dx')
        axes[1, 0].legend()
        import matplotlib.ticker as mticker
        axes[1, 0].xaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
        axes[1, 0].ticklabel_format(style='plain', axis='x')

        error = true_dydx[:] - pred_dydx[:]
        axes[1, 1].plot(error, label='Error (True - Pred)')
        axes[1, 1].set_title('Prediction Error (dim 1)')
        axes[1, 1].set_xlabel('Time Step')
        axes[1, 1].set_ylabel('Error')
        axes[1, 1].legend()
        axes[1, 1].xaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
        axes[1, 1].ticklabel_format(style='plain', axis='x')

        plt.tight_layout()
        plt.show()
