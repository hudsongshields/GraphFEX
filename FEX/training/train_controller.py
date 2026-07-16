from pathlib import Path

from ..utils.tree_configs import TreeConfig

from ..models.controllers import Controller
from ..models.learnable_tree import FEX
from ..models.nodes import Node
from ..utils.sampler import epsilon_greedy_sample
from .train_fex import train_network_fex, train_fex
from .train_configs import ControllerConfig, FEXConfig, runtimeconfig
from ..utils.pools import GraphPoolCandidate, GraphPool, Pool, PoolCandidate

from typing import Callable
import torch
import torch.multiprocessing as mp
import torch.optim.lr_scheduler as lr_scheduler
import os
import math
import time

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
    logger = train_logger
    if gpu_id is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        logger.info(f"Evaluating candidate {k_cand} on GPU {gpu_id}")
    else:
        device = torch.device("cpu")
        logger.info(f"Evaluating candidate {k_cand} on CPU")
    if k_cand == 0:
        verbose = True
        logger.info(f"Evaluating candidate {k_cand} with op indices: {op_indices}")
    else: verbose = False

    forcing_op_indices = op_indices[:len(self_ops_per_node)]
    inter_dynam_op_indices = None
    inter_fex = None

    if inter_ops_per_node is not None:
        inter_dynam_op_indices = op_indices[len(self_ops_per_node):len(self_ops_per_node) + len(inter_ops_per_node)]
        inter_fex = FEX(sample_indices=inter_dynam_op_indices, **inter_fex_kwargs).to(device)

    forcing_fex = FEX(sample_indices=forcing_op_indices, **fex_kwargs).to(device)

    if inter_ops_per_node is not None:
        score = train_network_fex(
            forcing_fex,
            inter_fex,
            dataloader_global,
            adj_matrix_global,
            config=fex_config_global,
            device=device,
            verbose=False,
        )
    else:
        score = train_fex(
            forcing_fex,
            dataloader_global,
            config=fex_config_global,
            device=device,
            verbose=False,
        )
    score = score.detach().item() if isinstance(score, torch.Tensor) else float(score)
    if not math.isfinite(score):
        reward = 0.0
    else:
        reward = 1.0 / math.sqrt(1.0 + score)

    for param in forcing_fex.parameters():
        param.requires_grad = False
        param.data = param.data.cpu()
    if inter_fex is not None:
        for param in inter_fex.parameters():
            param.requires_grad = False
            param.data = param.data.cpu()
        inter_fex = inter_fex.cpu()

    forcing_fex = forcing_fex.cpu()
    if k_cand == -1:
        logger.info(f"Sampled candidate {k_cand} with score {score:.4f}\n self dynamics: {forcing_fex}\n inter dynamics: {inter_fex}")
    return op_indices, reward, k_cand


def init_shared_resources(self_ops, inter_ops, fex_kwargs_input, inter_fex_kwargs_input, dataloader, adj_matrix, fex_config, logger_path=None):
    global self_ops_per_node, inter_ops_per_node, inter_fex_kwargs, fex_kwargs
    global dataloader_global, adj_matrix_global, fex_config_global

    self_ops_per_node = self_ops
    inter_ops_per_node = inter_ops
    fex_kwargs = fex_kwargs_input
    inter_fex_kwargs = inter_fex_kwargs_input
    
    dataloader_global = dataloader
    adj_matrix_global = adj_matrix
    fex_config_global = fex_config

    if logger_path:
        runtimeconfig.CreateLogger(logger_path, name="train_logger", mode="a")


def train_network_controller(self_fex_struct: TreeConfig, inter_fex_struct: TreeConfig, dataloader, adj_matrix, config: ControllerConfig, fex_config: FEXConfig, checkpoint_dir: Path = None, num_workers: int = 2) -> GraphPool:
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

    fex_kwargs = {
        "leaf_dim": next(iter(dataloader))[0].shape[2],
        "tree_structure": self_fex_struct,
    }
    fex_kwargs["num_leaves"] = self_fex_struct.num_leaves
    inter_fex_kwargs = {
        "leaf_dim": next(iter(dataloader))[0].shape[2] * 2,
        "tree_structure": inter_fex_struct,
    }
    inter_fex_kwargs["num_leaves"] = inter_fex_struct.num_leaves
    best_candidates = GraphPool(pool_size=config.poolsize)

    slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK")
    if slurm_cpus is not None:
        num_processes = int(slurm_cpus)
        train_logger.info(f"Detected SLURM environment with {num_processes} CPUs allocated for this task")
    else:
        num_processes = mp.cpu_count()
    if num_workers:
        num_processes = min(num_workers, num_processes)
    num_threads = min(num_processes, config.num_cands_per_epoch)
    print(f"Using {num_threads} parallel processes for candidate evaluation.")

    context = mp.get_context("spawn")
    gpu_ids = list(range(num_gpus)) if num_gpus > 0 else [None]
    with context.Pool(processes=num_threads, initializer=init_shared_resources, initargs=(self_ops_per_node, inter_ops_per_node, fex_kwargs, inter_fex_kwargs, dataloader_global, adj_matrix_global, fex_config_global, runtimeconfig.train_log_path)) as contextpool:
        for epoch in range(config.num_epochs):
            optimizer.zero_grad()
            num_cands = config.num_cands_per_epoch
            threshold = config.percentile_threshold
            thresh_idx = int(threshold * num_cands)
            log_probs = []
            pmfs = controller(controller_input)
            op_indices_list = []
            for k_cand in range(num_cands):
                op_indices = epsilon_greedy_sample(pmfs, epsilon=config.epsilon_greedy).squeeze(0)
                chosen_probs = torch.stack([pmfs[i].squeeze(0)[op_indices[i]] for i in range(len(op_indices))])
                log_prob = torch.log(chosen_probs).sum()
                log_probs.append(log_prob)
                op_indices_list.append(op_indices.detach().clone().cpu())
            top_epoch_cands = GraphPool(pool_size=thresh_idx)
            t1 = time.time()
            results = contextpool.starmap(eval_candidate, [(k_cand, gpu_ids[k_cand % len(gpu_ids)], op_indices_list[k_cand]) for k_cand in range(num_cands)])
            t2 = time.time()
            print(f"Evaluation time for fex epoch: {((t2 - t1) / num_cands / fex_config.num_epochs):.2f} seconds")
            for op_indices, reward, k_cand in results:
                inter_tree = FEX(sample_indices=op_indices[len(self_ops_per_node):], **inter_fex_kwargs)
                forcing_tree = FEX(sample_indices=op_indices[:len(self_ops_per_node)], **fex_kwargs)
                candidate = GraphPoolCandidate(inter_tree=inter_tree, forcing_tree=forcing_tree, reward=reward, id=int(k_cand + epoch * config.num_cands_per_epoch))
                top_epoch_cands.add_new(candidate)

            print(f"Epoch {epoch} pmfs: {[pmf.detach().cpu().numpy() for pmf in pmfs]}")

            rewards = torch.tensor([cand.reward for cand in top_epoch_cands]).to(runtimeconfig.device)
            log_probs_sorted = [log_probs[cand.id - epoch * config.num_cands_per_epoch] for cand in top_epoch_cands]

            thresh_reward = torch.tensor(top_epoch_cands.threshold).to(runtimeconfig.device)
            advantage = rewards - thresh_reward
            loss = -(advantage * torch.stack(log_probs_sorted)).mean()

            loss.backward()
            optimizer.step()

            for candidate in top_epoch_cands:
                best_candidates.add_new(candidate)
            if checkpoint_dir is not None:
                best_candidates.save_candidates(str(checkpoint_dir / "best_candidates"))
                best_candidates.visualize_candidates(str(checkpoint_dir / "visualizations"))


    return best_candidates

# for no inter tree, just self fex
def train_controller(self_fex_struct: TreeConfig, dataloader, controller_config: ControllerConfig, fex_config: FEXConfig, checkpoint_dir=None, num_workers:int = 1):
    train_logger = runtimeconfig.train_logger
    num_gpus = torch.cuda.device_count()
    
    global self_ops_per_node, fex_kwargs, dataloader_global, fex_config_global
    dataloader_global = dataloader
    fex_config_global = fex_config

    self_ops_per_node = self_fex_struct.ops_per_node
    controller = Controller(
        ops_per_node=self_ops_per_node,
        input_size=controller_config.input_dim,
        hidden_size=controller_config.hidden_dim,
    ).to(runtimeconfig.device)
    optimizer = torch.optim.Adam(controller.parameters(), lr=controller_config.lr)
    controller_input = torch.zeros(controller_config.input_dim).to(runtimeconfig.device)

    fex_kwargs = {
        "leaf_dim": next(iter(dataloader))[0].shape[-1],
        "tree_structure": self_fex_struct,
    }
    fex_kwargs["num_leaves"] = self_fex_struct.num_leaves
    best_candidates = Pool(pool_size=controller_config.poolsize)

    slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK")
    if slurm_cpus is not None:
        num_processes = int(slurm_cpus)
        train_logger.info(f"Detected SLURM environment with {num_processes} CPUs allocated for this task")
    else:
        num_processes = mp.cpu_count()
    if num_workers:
        num_processes = min(num_workers, num_processes)
    num_threads = min(num_processes, controller_config.num_cands_per_epoch)
    print(f"Using {num_threads} parallel processes for candidate evaluation.")

    context = mp.get_context("spawn")
    gpu_ids = list(range(num_gpus)) if num_gpus > 0 else [None]
    with context.Pool(processes=num_threads, initializer=init_shared_resources, initargs=(self_ops_per_node, None, fex_kwargs, None, dataloader_global, None, fex_config_global, runtimeconfig.train_log_path)) as contextpool:
        for epoch in range(controller_config.num_epochs):
            optimizer.zero_grad()
            num_cands = controller_config.num_cands_per_epoch
            threshold = controller_config.percentile_threshold
            thresh_idx = int(threshold * num_cands)
            log_probs = []
            pmfs = controller(controller_input)
            op_indices_list = []
            for k_cand in range(num_cands):
                op_indices = epsilon_greedy_sample(pmfs, epsilon=controller_config.epsilon_greedy).squeeze(0)
                chosen_probs = torch.stack([pmfs[i].squeeze(0)[op_indices[i]] for i in range(len(op_indices))])
                log_prob = torch.log(chosen_probs).sum()
                log_probs.append(log_prob)
                op_indices_list.append(op_indices.detach().clone().cpu())
            top_epoch_cands = Pool(pool_size=thresh_idx)
            t1 = time.time()
            results = contextpool.starmap(eval_candidate, [(k_cand, gpu_ids[k_cand % len(gpu_ids)], op_indices_list[k_cand]) for k_cand in range(num_cands)])
            t2 = time.time()
            print(f"Evaluation time for fex epoch: {((t2 - t1) / num_cands / fex_config.num_epochs):.2f} seconds")
            for op_indices, reward, k_cand in results:
                forcing_tree = FEX(sample_indices=op_indices, **fex_kwargs)
                candidate = PoolCandidate(tree=forcing_tree, reward=reward, id=int(k_cand + epoch * controller_config.num_cands_per_epoch))
                top_epoch_cands.add_new(candidate)

            print(f"Epoch {epoch}, pmfs: {[pmf.detach().cpu().numpy() for pmf in pmfs]}")

            rewards = torch.tensor([cand.reward for cand in top_epoch_cands]).to(runtimeconfig.device)
            log_probs_sorted = [log_probs[cand.id - epoch * controller_config.num_cands_per_epoch] for cand in top_epoch_cands]

            thresh_reward = torch.tensor(top_epoch_cands.threshold).to(runtimeconfig.device)
            advantage = rewards - thresh_reward
            loss = -(advantage * torch.stack(log_probs_sorted)).mean()

            loss.backward()
            optimizer.step()

            for candidate in top_epoch_cands:
                best_candidates.add_new(candidate)
            if checkpoint_dir is not None:
                best_candidates.save_candidates(str(checkpoint_dir / "best_candidates"))
                best_candidates.visualize_candidates(str(checkpoint_dir / "visualizations"))


    return best_candidates