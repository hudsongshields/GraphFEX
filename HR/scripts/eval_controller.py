import os
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from FEX.training.train_configs import FEXConfig, runtimeconfig
from FEX.training.train_controller import ControllerConfig, train_network_controller
from FEX.utils.numerical_deriv import NumericalDeriv
from FEX.utils.tree_configs import get_tree_config


SCRIPT_DIR = Path(__file__).resolve().parent
HR_DIR = SCRIPT_DIR.parent
DATA_DIR = HR_DIR / "data"


def setup_run_dir() -> Path:
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    run_dir = HR_DIR / "logs" / f"run_{job_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "pre_finetune").mkdir(parents=True, exist_ok=True)
    return run_dir


def load_hr_data():
    adj_path = DATA_DIR / "BA_Nnodes100_Adj_deg_7_1.csv"
    x_data_path = DATA_DIR / "HR_timeseries_SNR_40.csv"

    if not adj_path.exists():
        raise FileNotFoundError(f"Could not find adjacency matrix at: {adj_path}")
    if not x_data_path.exists():
        raise FileNotFoundError(f"Could not find timeseries data at: {x_data_path}")

    adj_matrix = pd.read_csv(adj_path, header=None)
    num_graph_nodes = adj_matrix.shape[0]

    x_df = pd.read_csv(x_data_path, header=None)
    num_timesteps, _ = x_df.shape

    x_np = x_df.to_numpy(dtype=np.float32)
    x_data = torch.from_numpy(x_np.reshape(num_timesteps, num_graph_nodes, 3))

    dt = 0.01
    len_run = 500
    per_run_timesteps = int(len_run / dt)
    cut_timestep = int(per_run_timesteps)

    num_runs = num_timesteps // per_run_timesteps
    x_chunks = torch.chunk(x_data, num_runs, dim=0)

    all_x = []
    all_dx_dt = []

    for x_run in x_chunks:
        x_run = x_run[::5]
        dx_dt = NumericalDeriv(x_run, dt=dt)
        x_run = x_run[2:-2]

        all_x.append(x_run)
        all_dx_dt.append(dx_dt)

    all_x = all_x[:]
    all_dx_dt = all_dx_dt[:]

    x_data = torch.cat(all_x, dim=0)
    dx_dt = torch.cat(all_dx_dt, dim=0)

    print(f"num runs: {num_runs}, timesteps per run: {per_run_timesteps}, total timesteps: {x_data.shape[0]}")
    adj_matrix_tensor = torch.tensor(adj_matrix.values, dtype=torch.float32)

    return x_data, dx_dt, adj_matrix_tensor


def make_dataloader(x_data, dx_dt):
    dataset = TensorDataset(x_data, dx_dt)

    pin_memory = torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=512,
        shuffle=True,
        pin_memory=pin_memory,
    )


def main():
    run_dir = setup_run_dir()

    log_path = run_dir / "controller_eval.log"
    runtimeconfig.CreateLogger(str(log_path), name="train_logger")

    forcing_tree_config = get_tree_config("depth_3_leaves_4_config")
    inter_tree_config = get_tree_config("depth_2_tree_config")

    x_data, dx_dt, adj_matrix_tensor = load_hr_data()
    dataloader = make_dataloader(x_data, dx_dt)

    controller_config = ControllerConfig(
        input_dim=20,
        hidden_dim=64,
        lr=0.001,
        num_epochs=800,
        num_cands_per_epoch=12,
        percentile_threshold=0.5,
        num_trees=2,
    )

    fex_config = FEXConfig(
        num_epochs=60,
        bfgs_epochs=0,
        bfgs_lr=0.1,
        leaf_dim=x_data.shape[2],
        num_leaves=forcing_tree_config.num_leaves,
        weight_decay=0.0,
        mag_entropy_weight=1e-4,
        pct_cosine_restart=0.5,
        tau_start=8.0,
        tau_end=4.0,
    )

    save_dir = run_dir / "pre_finetune"
    best_candidates = train_network_controller(
        forcing_tree_config,
        inter_tree_config,
        dataloader,
        adj_matrix_tensor,
        controller_config,
        fex_config,
        checkpoint_dir=str(save_dir / "best_candidates.pt"),
    )
    best_candidates.save_candidates(str(save_dir / "best_candidates.pt"))
    best_candidates.visualize_candidates(
        str(save_dir / "candidate_viz"),
        clear_directory=True,
    )


if __name__ == "__main__":
    main()
