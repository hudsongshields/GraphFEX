import argparse
import os
import multiprocessing as mp
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from FEX.training.train_configs import FEXConfig
from FEX.training.train_controller import ControllerConfig, train_network_controller
from FEX.helpers.tree_configs import get_tree_config
from NumericalExperiments.HR.generate_data import make_adjacency, make_timeseries

"""This is a temporary script"""

SCRIPT_DIR = Path(__file__).resolve().parent
HR_DIR = SCRIPT_DIR.parent
DATA_DIR = HR_DIR / "data"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--snr", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=80)
    parser.add_argument("--controller_epochs", type=int, default=400)
    parser.add_argument("--num_stoch_batches", type=int, default=1)

    parser.add_argument("--pct_edges", type=float, default=0.1)

    parser.add_argument("--num_cands_per_epoch", type=int, default=10)
    args = parser.parse_args()

    seed_everything(args.seed)

    run_dir = setup_run_dir()
    save_dir = run_dir / f"pre_finetune_dim_0_snr{args.snr if args.snr is not None else 'None'}"
    save_dir.mkdir(parents=True, exist_ok=True)

    log_path = save_dir / "controller_eval.log"

    forcing_tree_config = get_tree_config("depth_3_leaves_4_config")
    inter_tree_config = get_tree_config("depth_2_tree_config")

    adjacency = make_adjacency(args.nodes, probability=args.pct_edges, device=device)
    states, derivatives = make_timeseries(args.samples, adjacency, snr=args.snr)
    dataloader = DataLoader(
        TensorDataset(states, derivatives),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=device == "cuda",
    )

    controller_config = ControllerConfig(
        input_dim=20,
        hidden_dim=64,
        lr=0.003,
        num_epochs=args.controller_epochs,
        num_cands_per_epoch=args.num_cands_per_epoch,
        percentile_threshold=0.5,
        epsilon_greedy=0.2
    )

    fex_config = FEXConfig(
        num_epochs=args.num_epochs,
        bfgs_epochs=0,
        lr=0.2,
        inter_lr=0.2,
        bfgs_lr=0.1,


        target_dim=0
    )

    best_candidates = train_network_controller(
        forcing_tree_config,
        inter_tree_config,
        dataloader,
        adjacency,
        controller_config,
        fex_config,
        checkpoint_dir=save_dir,
        num_workers=args.num_workers,
        num_groups=args.num_stoch_batches,
    )
    best_candidates.save_candidates(str(save_dir / "best_candidates.pt"))
    best_candidates.visualize_candidates(
        str(save_dir / "candidate_viz"),
        clear_directory=True,
    )


if __name__ == "__main__":
    main()