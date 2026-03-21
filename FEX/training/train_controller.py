from ..utils.tree_configs import TreeConfig

from ..models.controllers import Controller
from ..models.learnable_tree import FEX
from ..models.nodes import Node
from ..utils.sampler import epsilon_greedy_sample
from .train_fex import train_network_fex
from .train_configs import ControllerConfig, FEXConfig, runtimeconfig
from ..utils.pools import GraphPoolCandidate, GraphPool

from typing import Callable
import torch
import torch.optim.lr_scheduler as lr_scheduler
import math

import logging
train_logger = logging.getLogger("train_logger") 
train_logger.setLevel(logging.INFO)


def train_network_controller(tree_structure: TreeConfig, dataloader, adj_matrix, config: ControllerConfig, fex_config: FEXConfig):
    train_logger = runtimeconfig.train_logger
    controller = Controller(
        ops_per_node=tree_structure.ops_per_node,
        num_trees=config.num_trees,
        input_size=config.input_dim,
        hidden_size=config.hidden_dim,
    ).to(runtimeconfig.device)
    
    optimizer = torch.optim.Adam(controller.parameters(), lr=config.lr)
    controller_input = torch.zeros(config.input_dim).to(runtimeconfig.device)
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.num_epochs,
        eta_min=max(1e-5, config.lr * 0.005),
    )
    
    fex_kwargs = {
        "leaf_dim": fex_config.leaf_dim,
        "num_leaves": fex_config.num_leaves,
        "tree_structure": tree_structure,
    }
    inter_fex_kwargs = fex_kwargs.copy()
    inter_fex_kwargs["leaf_dim"] = fex_config.leaf_dim * 2

    adj_matrix = adj_matrix.to(runtimeconfig.device)

    best_candidates = GraphPool(pool_size=5)
    for epoch in range(config.num_epochs):
        optimizer.zero_grad()
        num_cands = config.num_cands_per_epoch
        threshold = config.percentile_threshold
        thresh_idx = int(threshold * num_cands)
        log_probs = []
        top_epoch_cands = GraphPool(pool_size=thresh_idx)
        pmfs = controller(controller_input)

        for k_cand in range(num_cands):
            op_indices = epsilon_greedy_sample(pmfs, epsilon=0.1).squeeze(0) # (num_nodes,)
            chosen_probs = torch.stack([pmfs[i].squeeze(0)[op_indices[i]] for i in range(len(op_indices))])
            log_prob = torch.log(chosen_probs).sum()
            log_probs.append(log_prob)

            # Split sampled indices to create seperate sequences for each FEX (forcing and interaction dynamics)
            inter_dynam_op_indices = op_indices[len(op_indices) // 2:]
            forcing_op_indices = op_indices[:len(op_indices) // 2]

            inter_fex = FEX(sample_indices=inter_dynam_op_indices, **inter_fex_kwargs).to(runtimeconfig.device)
            forcing_fex = FEX(sample_indices=forcing_op_indices, **fex_kwargs).to(runtimeconfig.device)
            score = train_network_fex(
                forcing_fex,
                inter_fex,
                dataloader,
                adj_matrix,
                config=fex_config,
            )
            score = score.detach().item() if isinstance(score, torch.Tensor) else float(score)
            if not math.isfinite(score):
                reward = 0.0
                train_logger.warning("Epoch %s, Candidate %s produced non-finite score; assigning zero reward.", epoch, k_cand,)
            else:
                reward = 1.0 / math.sqrt(1.0 + score)
            candidate = GraphPoolCandidate(inter_tree=inter_fex, forcing_tree=forcing_fex, reward=reward, id=k_cand)
            top_epoch_cands.add_new(candidate)
            train_logger.info(f"Epoch {epoch}, Candidate {k_cand}, Score: {score:.4f}, Reward: {reward:.4f}")
        train_logger.debug(f"Epoch {epoch} pmfs: {[pmf.detach().cpu().numpy() for pmf in pmfs]}")

        rewards = torch.tensor([cand.reward for cand in top_epoch_cands]).to(runtimeconfig.device)
        log_probs_sorted = [log_probs[cand.id] for cand in top_epoch_cands]

        thresh_reward = torch.tensor(top_epoch_cands.threshold).to(runtimeconfig.device)
        advantage = rewards - thresh_reward
        # advantage = torch.softmax(advantage / (advantage.std() + 1e-8), dim=0) # Normalize advantages for stability
        """with torch.no_grad():
            advantage = torch.softmax(advantage / (advantage.std() + 1e-8), dim=0)"""
        loss = -(advantage * torch.stack(log_probs_sorted)).mean()

        loss.backward()
        optimizer.step()
        scheduler.step()


        for candidate in top_epoch_cands:
            best_candidates.add_new(candidate)

        
        train_logger.info(f"Controller Epoch {epoch}, Loss: {loss.item()}")
        train_logger.debug(f"Epoch {epoch}, Reward Threshold for Backprop: {thresh_reward:.4f}")
        train_logger.debug(f"Epoch {epoch}, Rewards: {rewards.detach().cpu().numpy()}")
        train_logger.debug(f"Epoch {epoch}, Log Probs: {[lp.detach().cpu().numpy() for lp in log_probs_sorted]}")
        train_logger.debug(f"Epoch {epoch}, Advantage: {advantage.detach().cpu().numpy()}")

    return best_candidates


if __name__ == "__main__":
    from torch.utils.data import DataLoader, TensorDataset
    from ..utils.tree_configs import depth_2_tree
    from ..utils.operations import UNARY_OPS, BINARY_OPS


    # fake data and adjacency matrix for testing
    num_nodes = 100
    num_timesteps = 10
    x_data = torch.randn(num_timesteps, num_nodes)
    dx_dt = torch.randn(num_timesteps, num_nodes)
    adj_matrix = torch.randint(0, 2, (num_nodes, num_nodes)).float()

    dataloader = DataLoader(TensorDataset(x_data, dx_dt), batch_size=32, shuffle=True)

    input_dim = 20 # Controller input dim is arbitrary
    num_epochs = 10
    learning_rate = 0.001
    controller_config = ControllerConfig(input_dim=input_dim, num_epochs=num_epochs, lr=learning_rate, num_trees=2)

    tree_epochs = 5
    tree_lr = 0.001
    fex_config = FEXConfig(num_epochs=tree_epochs, lr=tree_lr, max_nodes=adj_matrix.size(0))


    num_unary_ops = len(UNARY_OPS)
    num_binary_ops = len(BINARY_OPS)
    tree_type = "depth_2"
    if tree_type == "depth_2":
        parent_node = depth_2_tree([0, 1, 2])
        ops_per_node = []
        leaf_nodes = 0
        for node in parent_node.preorder_traversal():
            if node.operation_type == "leaf":
                leaf_nodes += 1
            elif node.operation_type == "unary":
                ops_per_node.append(num_unary_ops)
            elif node.operation_type == "binary":
                ops_per_node.append(num_binary_ops)

    fex_config.num_leaves = leaf_nodes
    fex_config.leaf_dim = 1

    controller = Controller(input_size=input_dim, ops_per_node=ops_per_node, num_trees=controller_config.num_trees, hidden_size=20).to(runtimeconfig.device)
    train_network_controller(controller, depth_2_tree, dataloader, adj_matrix, controller_config, fex_config)

