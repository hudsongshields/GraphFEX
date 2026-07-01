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
from FEX.training.train_controller import ControllerConfig, train_controller
from FEX.utils.numerical_deriv import NumericalDeriv
from FEX.utils.tree_configs import get_tree_config
from HR.data.generate_data import make_adjacency, make_data


SCRIPT_DIR = Path(__file__).resolve().parent
HR_DIR = SCRIPT_DIR.parent
DATA_DIR = HR_DIR / "data"

def setup_run_dir() -> Path:
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    run_dir = HR_DIR / "logs_controller" / f"run_{job_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--nodes", type=int, default=100)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dim", type=int, default=1)
    parser.add_argument("--snr", type=float, default=None)
    parser.add_argument("--num_epochs", type=int, default=60)
    args = parser.parse_args()

    seed_everything(args.seed)

    run_dir = setup_run_dir()

    log_path = run_dir / "controller_eval.log"
    runtimeconfig.CreateLogger(str(log_path), name="train_logger")

    forcing_tree_config = get_tree_config("depth_2_tree_config")

    adjacency = make_adjacency(args.nodes, probability=0.35, device=runtimeconfig.device)
    states, derivatives = make_data(args.samples, adjacency, snr=args.snr)
    dataloader = DataLoader(
        TensorDataset(states, derivatives),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=runtimeconfig.device == "cuda",
    )

    controller_config = ControllerConfig(
        input_dim=20,
        hidden_dim=64,
        lr=0.005,
        num_epochs=200,
        num_cands_per_epoch=10,
        percentile_threshold=0.4,
        num_trees=2,
        epsilon_greedy=0.2
    )

    fex_config = FEXConfig(
        num_epochs=args.num_epochs,
        bfgs_epochs=0,
        lr=0.2,
        bfgs_lr=0.1,
        leaf_dim=states.shape[2],
        num_leaves=forcing_tree_config.num_leaves,

        target_dim=args.dim,
    )

    save_dir = run_dir / f"pre_finetune_dim{args.dim}_snr{args.snr if args.snr is not None else 'None'}"
    save_dir.mkdir(parents=True, exist_ok=True)
    best_candidates = train_controller(
        forcing_tree_config,
        dataloader,
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
