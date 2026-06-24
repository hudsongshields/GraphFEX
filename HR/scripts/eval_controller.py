import argparse
import os
import multiprocessing as mp
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from FEX.training.train_configs import FEXConfig, runtimeconfig
from FEX.training.train_controller import ControllerConfig, train_network_controller
from FEX.utils.numerical_deriv import NumericalDeriv
from FEX.utils.tree_configs import get_tree_config
from HR.data.generate_data import make_adjacency, make_data


SCRIPT_DIR = Path(__file__).resolve().parent
HR_DIR = SCRIPT_DIR.parent
DATA_DIR = HR_DIR / "data"

def setup_run_dir() -> Path:
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    run_dir = HR_DIR / "sigmoid_eval" / f"run_{job_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "pre_finetune").mkdir(parents=True, exist_ok=True)
    return run_dir

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    seed_everything(42) # for reproducibility

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--nodes", type=int, default=100)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    run_dir = setup_run_dir()

    log_path = run_dir / "controller_eval.log"
    runtimeconfig.CreateLogger(str(log_path), name="train_logger")

    forcing_tree_config = get_tree_config("depth_3_leaves_4_config")
    inter_tree_config = get_tree_config("depth_2_tree_config")

    adjacency = make_adjacency(args.nodes, probability=0.35, device=runtimeconfig.device)
    states, derivatives = make_data(args.samples, adjacency)
    dataloader = DataLoader(
        TensorDataset(states, derivatives),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=runtimeconfig.device == "cuda",
    )

    controller_config = ControllerConfig(
        input_dim=20,
        hidden_dim=64,
        lr=0.001,
        num_epochs=200,
        num_cands_per_epoch=10,
        percentile_threshold=0.2,
        num_trees=2,
        epsilon_greedy=0.3
    )

    fex_config = FEXConfig(
        num_epochs=80,
        bfgs_epochs=40,
        lr=0.2,
        inter_lr=0.2,
        bfgs_lr=0.1,
        leaf_dim=states.shape[2],
        num_leaves=forcing_tree_config.num_leaves,
        mag_entropy_weight=0,
        pct_cosine_restart=1.0,
        tau_start=1.0,
        tau_end=1.0,
    )

    save_dir = run_dir / "pre_finetune"
    best_candidates = train_network_controller(
        forcing_tree_config,
        inter_tree_config,
        dataloader,
        adjacency,
        controller_config,
        fex_config,
        checkpoint_dir=save_dir,
        num_workers=args.num_workers,
    )
    best_candidates.save_candidates(str(save_dir / "best_candidates.pt"))
    best_candidates.visualize_candidates(
        str(save_dir / "candidate_viz"),
        clear_directory=True,
    )


if __name__ == "__main__":
    main()
