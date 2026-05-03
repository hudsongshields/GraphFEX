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
import torch.multiprocessing as mp
from multiprocessing.managers import BaseManager
import torch.optim.lr_scheduler as lr_scheduler
import os
import math

import logging
train_logger = logging.getLogger("train_logger") 
train_logger.setLevel(logging.INFO)

self_ops_per_node = None
inter_ops_per_node = None
inter_fex_kwargs = None
fex_kwargs = None

dataloader_global = None
adj_matrix_global = None
fex_config_global = None


def eval_candidate(k_cand, gpu_id, op_indices):
    if gpu_id is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")

    forcing_op_indices = op_indices[:len(self_ops_per_node)]
    inter_dynam_op_indices = op_indices[len(self_ops_per_node):len(self_ops_per_node) + len(inter_ops_per_node)]

    inter_fex = FEX(sample_indices=inter_dynam_op_indices, **inter_fex_kwargs).to(device)
    forcing_fex = FEX(sample_indices=forcing_op_indices, **fex_kwargs).to(device)

    score = train_network_fex(
        forcing_fex,
        inter_fex,
        dataloader_global,
        adj_matrix_global,
        config=fex_config_global,
        device=device,
    )
    score = score.detach().item() if isinstance(score, torch.Tensor) else float(score)
    if not math.isfinite(score):
        reward = 0.0
    else:
        reward = 1.0 / math.sqrt(1.0 + score)

    for param in forcing_fex.parameters():
        param.requires_grad = False
        param.data = param.data.cpu()
    for param in inter_fex.parameters():
        param.requires_grad = False
        param.data = param.data.cpu()
    
    inter_fex = inter_fex.cpu()
    forcing_fex = forcing_fex.cpu()
    if k_cand == 0:
        train_logger.info(f"Sampled candidate {k_cand} with score {score:.4f}\n self dynamics: {forcing_fex}\n inter dynamics: {inter_fex}")
    return op_indices, reward, k_cand


def init_shared_resources(self_ops, inter_ops, fex_kwargs_input, inter_fex_kwargs_input, dataloader, adj_matrix, fex_config):
    global self_ops_per_node, inter_ops_per_node, inter_fex_kwargs, fex_kwargs
    global dataloader_global, adj_matrix_global, fex_config_global

    self_ops_per_node = self_ops
    inter_ops_per_node = inter_ops
    fex_kwargs = fex_kwargs_input
    inter_fex_kwargs = inter_fex_kwargs_input
    
    dataloader_global = dataloader
    adj_matrix_global = adj_matrix
    fex_config_global = fex_config


def train_network_controller(self_fex_struct: TreeConfig, inter_fex_struct: TreeConfig, dataloader, adj_matrix, config: ControllerConfig, fex_config: FEXConfig):
    train_logger = runtimeconfig.train_logger
    num_gpus = torch.cuda.device_count()
    
    global self_ops_per_node, inter_ops_per_node, inter_fex_kwargs, fex_kwargs, dataloader_global, adj_matrix_global, fex_config_global
    dataloader_global = dataloader
    adj_matrix_global = adj_matrix
    fex_config_global = fex_config

    self_ops_per_node = self_fex_struct.ops_per_node
    inter_ops_per_node = inter_fex_struct.ops_per_node
    controller = Controller(
        ops_per_node=self_ops_per_node + inter_ops_per_node,
        input_size=config.input_dim,
        hidden_size=config.hidden_dim,
    ).to(runtimeconfig.device)
    optimizer = torch.optim.Adam(controller.parameters(), lr=config.lr)
    controller_input = torch.zeros(config.input_dim).to(runtimeconfig.device)
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.num_epochs,
        eta_min=max(1e-5, config.lr * 0.05),
    )
    fex_kwargs = {
        "leaf_dim": fex_config.leaf_dim,
        "tree_structure": self_fex_struct,
    }
    fex_kwargs["num_leaves"] = self_fex_struct.num_leaves
    inter_fex_kwargs = {
        "leaf_dim": fex_config.leaf_dim * 2,
        "tree_structure": inter_fex_struct,
    }
    inter_fex_kwargs["num_leaves"] = inter_fex_struct.num_leaves
    best_candidates = GraphPool(pool_size=10)

    slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK")
    if slurm_cpus is not None:
        num_processes = int(slurm_cpus)
        train_logger.info(f"Detected SLURM environment with {num_processes} CPUs allocated for this task. Using {num_processes} parallel processes for candidate evaluation.")
    else: num_processes = mp.cpu_count()
    num_threads = min(mp.cpu_count(), config.num_cands_per_epoch) if num_processes is None else num_processes
    print(f"Using {num_threads} parallel processes for candidate evaluation (hardware threads: {mp.cpu_count()})")

    for epoch in range(config.num_epochs):
        optimizer.zero_grad()
        num_cands = config.num_cands_per_epoch
        threshold = config.percentile_threshold
        thresh_idx = int(threshold * num_cands)
        log_probs = []
        pmfs = controller(controller_input)
        op_indices_list =[]
        for k_cand in range(num_cands):
            op_indices = epsilon_greedy_sample(pmfs, epsilon=0.1).squeeze(0)
            chosen_probs = torch.stack([pmfs[i].squeeze(0)[op_indices[i]] for i in range(len(op_indices))])
            log_prob = torch.log(chosen_probs).sum()
            log_probs.append(log_prob)
            
            op_indices_list.append(op_indices.detach().clone().cpu())

        top_epoch_cands = GraphPool(pool_size=thresh_idx)

        context = mp.get_context("spawn")
        gpu_ids = list(range(num_gpus)) if num_gpus > 0 else [None]
        with context.Pool(processes=num_threads, initializer=init_shared_resources, initargs=(self_ops_per_node, inter_ops_per_node, fex_kwargs, inter_fex_kwargs, dataloader_global, adj_matrix_global, fex_config_global)) as pool:
            results = pool.starmap(eval_candidate, [(k_cand, gpu_ids[k_cand % len(gpu_ids)], op_indices_list[k_cand]) for k_cand in range(num_cands)])

        for op_indices, reward, k_cand in results:
            inter_tree = FEX(sample_indices=op_indices[len(self_ops_per_node):len(self_ops_per_node) + len(inter_ops_per_node)], **inter_fex_kwargs)
            forcing_tree = FEX(sample_indices=op_indices[:len(self_ops_per_node)], **fex_kwargs)
            candidate = GraphPoolCandidate(inter_tree=inter_tree, forcing_tree=forcing_tree, reward=reward, id=k_cand)
            top_epoch_cands.add_new(candidate)

        train_logger.debug(f"Epoch {epoch} pmfs: {[pmf.detach().cpu().numpy() for pmf in pmfs]}")

        rewards = torch.tensor([cand.reward for cand in top_epoch_cands]).to(runtimeconfig.device)
        log_probs_sorted = [log_probs[cand.id] for cand in top_epoch_cands]

        thresh_reward = torch.tensor(top_epoch_cands.threshold).to(runtimeconfig.device)
        advantage = rewards - thresh_reward
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
    controller_config = ControllerConfig(input_dim=input_dim, num_epochs=num_epochs, lr=learning_rate)

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

