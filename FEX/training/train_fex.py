from ..models.learnable_tree import FEX
from .train_configs import FEXConfig, runtimeconfig
import torch
import math

def train_network_fex(forcing_tree: FEX, inter_dynam_tree: FEX, dataloader, adj_matrix, config: FEXConfig):
    adam_optim_forcing = torch.optim.Adam(forcing_tree.all_parameters(), lr=config.lr)
    adam_optim_inter = torch.optim.Adam(inter_dynam_tree.all_parameters(), lr=config.lr)
    train_logger = runtimeconfig.train_logger
    train_logger.debug(f"starting param forcing tree: {[param.detach().cpu().numpy() for param in forcing_tree.tree_params()]}")
    train_logger.debug(f"starting param inter tree: {[param.detach().cpu().numpy() for param in inter_dynam_tree.tree_params()]}")

    criterion = torch.nn.MSELoss()

    max_nodes = config.max_nodes
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
                interaction_output = 0

                neighbors = torch.nonzero(sparse_adj_matrix[node], as_tuple=False).flatten()
                for neighbor in neighbors:
                    neighbor = neighbor.item()
                    if neighbor == node:
                        continue
                    neighbor_input = torch.cat((node_input, sparse_batch_x[:, neighbor, :]), dim=-1)
                    interaction_output += sparse_adj_matrix[node, neighbor] * inter_dynam_tree(neighbor_input)

                node_outputs.append(forcing_output + interaction_output)

            batch_dy = torch.stack(node_outputs, dim=1)
            batch_loss = criterion(batch_dy, sparse_dx_dt_val)
                
                    
            batch_loss.backward()
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

    # prebuild train set for LBFGS closure
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
                interaction_output = 0


                neighbors = torch.nonzero(sparse_adj_matrix[node], as_tuple=False).flatten()
                for neighbor in neighbors:
                    neighbor = neighbor.item()
                    if neighbor == node:
                        continue
                    neighbor_input = torch.cat((node_input, sparse_batch_x[:, neighbor, :]), dim=-1)
                    interaction_output += sparse_adj_matrix[node, neighbor] * inter_dynam_tree(neighbor_input)

                node_outputs.append(forcing_output + interaction_output)

            batch_dy = torch.stack(node_outputs, dim=1)
            batch_loss = criterion(batch_dy, sparse_dx_dt_val)
            total_loss.append(batch_loss)

        total_loss = torch.stack(total_loss).mean()
        total_loss.backward()
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


if __name__ == "__main__":
    from torch.utils.data import DataLoader, TensorDataset
    from ..utils.tree_configs import get_tree_config
    from ..utils.operations import UNARY_OPS, BINARY_OPS
    import pandas as pd
    import numpy as np
    from ..utils.numerical_deriv import NumericalDeriv

    adj_matrix = pd.read_csv('HR/data/BA_Nnodes100_Adj.csv', header=None)
    num_graph_nodes = adj_matrix.shape[0]

    x_df = pd.read_csv('HR/data/HR_timeseries_SNR_40.csv', header=None)
    num_timesteps, num_cols = x_df.shape
    x_np = x_df.to_numpy(dtype=np.float32)
    x_data = torch.from_numpy(x_np.reshape(num_timesteps, num_graph_nodes, 3))

    dt = 0.01
    dx_dt = NumericalDeriv(x_data, dt=dt) # 4th order
    x_data = x_data[2:-2]


    train_test_split = int(0.8 * len(x_data))
    train_x_data = x_data[:train_test_split]
    train_dx_dt = dx_dt[:train_test_split]
    adj_matrix_tensor = torch.tensor(adj_matrix.values, dtype=torch.float32).to(runtimeconfig.device)
    x_data_tensor_ds = TensorDataset(train_x_data, train_dx_dt)
    x_data_tensor_ds = x_data_tensor_ds[:len(x_data_tensor_ds)//5]
    if runtimeconfig.device == "cuda":
        dataloader = DataLoader(x_data_tensor_ds, batch_size=32, shuffle=True, pin_memory=True)
    else:
        dataloader = DataLoader(x_data_tensor_ds, batch_size=32, shuffle=True)

    log_path = "FEX/training/testing/ground_truth_test.log"
    runtimeconfig.CreateLogger(log_path, name="train_logger")

    tree_config = get_tree_config("depth_3_leaves_4_config")
    fex_config = FEXConfig(
        num_epochs = 100, 
        lr = 0.002,
        bfgs_lr = 0.1,
        max_nodes = 20,
        max_norm = 1.0,
        leaf_dim = x_data.shape[2],

        num_leaves = tree_config.num_leaves
    )


    fex_kwargs = {
        "leaf_dim": fex_config.leaf_dim,
        "num_leaves": fex_config.num_leaves,
        "tree_structure": tree_config,
    }
    forcing_op_indices = torch.tensor([0, 2, 2, 0, 0, 1, 2], dtype=torch.long).to(runtimeconfig.device)
    forcing_fex = FEX(sample_indices=forcing_op_indices, **fex_kwargs).to(runtimeconfig.device)
    forcing_fex.visualize_tree("FEX/training/testing/initial_forcing_tree.png")

    inter_fex_kwargs = fex_kwargs.copy()
    inter_fex_kwargs["leaf_dim"] = fex_config.leaf_dim * 2
    inter_op_indices = torch.tensor([2, 0, 1, 5, 5, 0, 5], dtype=torch.long).to(runtimeconfig.device)
    inter_fex = FEX(sample_indices=inter_op_indices, **inter_fex_kwargs).to(runtimeconfig.device)
    inter_fex.visualize_tree("FEX/training/testing/initial_inter_tree.png")

    train_network_fex(forcing_fex, inter_fex, dataloader, adj_matrix_tensor, fex_config)
    forcing_fex.visualize_tree("FEX/training/testing/final_forcing_tree.png")
    inter_fex.visualize_tree("FEX/training/testing/final_inter_tree.png")




    