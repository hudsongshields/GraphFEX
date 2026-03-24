import torch
from torch import nn
from .train_configs import runtimeconfig, FEXConfig
from ..models.learnable_tree import FEX

def finetune_network_fex(forcing_tree: FEX, inter_dynam_tree: FEX, dataloader, adj_matrix, config: FEXConfig):
    adam_optim_forcing = torch.optim.Adam(forcing_tree.all_parameters(), lr=config.lr)
    adam_optim_inter = torch.optim.Adam(inter_dynam_tree.all_parameters(), lr=config.lr)
    scheduler_forcing = torch.optim.lr_scheduler.CosineAnnealingLR(adam_optim_forcing, T_max=(config.num_epochs * 0.8), eta_min=(config.lr * 0.05))
    scheduler_inter = torch.optim.lr_scheduler.CosineAnnealingLR(adam_optim_inter, T_max=(config.num_epochs * 0.8), eta_min=(config.lr * 0.05))

    train_logger = runtimeconfig.train_logger
    criterion = nn.MSELoss()
    max_nodes = config.max_nodes
    
    for epoch in range(config.num_epochs):
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
                interaction_count = 0

                neighbors = torch.nonzero(sparse_adj_matrix[node], as_tuple=False).flatten()
                for neighbor in neighbors:
                    neighbor = neighbor.item()
                    if neighbor == node:
                        continue
                    interaction_count += 1
                    neighbor_input = torch.cat((node_input, sparse_batch_x[:, neighbor, :]), dim=-1)
                    interaction_output += inter_dynam_tree(neighbor_input)

                node_outputs.append(forcing_output + (interaction_output/max(1, interaction_count)))    

            batch_dy = torch.stack(node_outputs, dim=1)
            batch_loss = criterion(batch_dy, sparse_dx_dt_val)
                
            batch_loss.backward()
            adam_optim_forcing.step()
            adam_optim_inter.step()

            scheduler_forcing.step()
            scheduler_inter.step()

            epoch_loss += batch_loss.detach().item()
        train_logger.info(f"Epoch {epoch}, Loss: {epoch_loss / len(dataloader)}")


if __name__ == "__main__":
    from torch.utils.data import DataLoader, TensorDataset
    from ..utils.tree_configs import get_tree_config
    from ..utils.operations import UNARY_OPS, BINARY_OPS
    import pandas as pd
    import numpy as np
    from ..utils.numerical_deriv import NumericalDeriv
    from .train_fex import train_network_fex

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
    if runtimeconfig.device == "cuda":
        dataloader = DataLoader(x_data_tensor_ds, batch_size=256, shuffle=True, pin_memory=True)
    else:
        dataloader = DataLoader(x_data_tensor_ds, batch_size=256, shuffle=True)

    log_path = "FEX/training/testing/ground_truth_test.log"
    runtimeconfig.CreateLogger(log_path, name="train_logger")

    tree_config = get_tree_config("depth_3_partial_config")
    fex_config = FEXConfig(
        num_epochs = 20, 
        lr = 0.002,
        bfgs_lr = 0.1,
        max_nodes = 20,
        leaf_dim = x_data.shape[2],

        num_leaves = tree_config.num_leaves
    )
    fex_kwargs = {
        "leaf_dim": fex_config.leaf_dim,
        "num_leaves": fex_config.num_leaves,
        "tree_structure": tree_config,
    }
    forcing_op_indices = torch.tensor([2, 0, 2, 1, 0], dtype=torch.long).to(runtimeconfig.device)
    forcing_fex = FEX(sample_indices=forcing_op_indices, **fex_kwargs).to(runtimeconfig.device)

    inter_fex_kwargs = fex_kwargs.copy()
    inter_fex_kwargs["leaf_dim"] = fex_config.leaf_dim * 2
    inter_op_indices = torch.tensor([1, 2, 5, 0, 0], dtype=torch.long).to(runtimeconfig.device)
    inter_fex = FEX(sample_indices=inter_op_indices, **inter_fex_kwargs).to(runtimeconfig.device)

    train_network_fex(forcing_fex, inter_fex, dataloader, adj_matrix_tensor, fex_config)
    forcing_fex.visualize_tree("FEX/training/testing/initial_forcing_tree.png")
    inter_fex.visualize_tree("FEX/training/testing/initial_inter_tree.png")

    fex_finetune_config = FEXConfig(
        num_epochs = 100,
        lr = 0.0002,
        max_nodes = 100,

        leaf_dim = x_data.shape[2],
        num_leaves = tree_config.num_leaves
    )
    finetune_network_fex(forcing_fex, inter_fex, dataloader, adj_matrix_tensor, fex_finetune_config)
    forcing_fex.visualize_tree("FEX/training/testing/final_forcing_tree.png")
    inter_fex.visualize_tree("FEX/training/testing/final_inter_tree.png")