import sys
from pathlib import Path

# resolve imports from the current working dir.
cwd = Path.cwd().resolve()
project_root = cwd.parent
sys.path.insert(0, str(project_root))

from FEX.utils.numerical_deriv import NumericalDeriv
from FEX.training.train_controller import ControllerConfig, train_network_controller
from FEX.training.train_configs import FEXConfig
from FEX.training.train_configs import runtimeconfig
from FEX.utils.tree_configs import get_tree_config

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from FEX.utils.tree_configs import *

# Initialize logger before training
from pathlib import Path
import os
PROJECT_ROOT = Path(__file__).resolve().parent.parent

job_id = os.environ.get("SLURM_JOB_ID", "local")
run_dir = PROJECT_ROOT / "logs" / f"run_{job_id}"
run_dir.mkdir(parents=True, exist_ok=True)

log_path = str(run_dir / "controller_eval.log")
runtimeconfig.CreateLogger(log_path, name="train_logger")

forcing_tree_config = get_tree_config("depth_3_leaves_4_config")
inter_tree_config = get_tree_config("depth_2_tree_config")

import pandas as pd
import numpy as np

# Update file paths to match train_fex.py
adj_matrix = pd.read_csv('../data/BA_Nnodes100_Adj_deg_7_1.csv', header=None)
num_graph_nodes = adj_matrix.shape[0]

x_df = pd.read_csv('../data/HR_timeseries_BA_deg_7_1_SNR_45.csv', header=None)
num_timesteps, num_cols = x_df.shape
x_np = x_df.to_numpy(dtype=np.float32)
x_data = torch.from_numpy(x_np.reshape(num_timesteps, num_graph_nodes, 3))

dt = 0.01
len_run = 500
per_run_timesteps = int(len_run / dt)
cut_timestep = int(per_run_timesteps * 0.1)


num_runs = num_timesteps // per_run_timesteps
x_chunks = torch.chunk(x_data, num_runs, dim=0)
all_dx_dt = []
all_x = []

for x_run in x_chunks:
    x_run = x_run[:cut_timestep]
    dx_dt = NumericalDeriv(x_run, dt=dt) # 4th order
    x_run = x_run[2:-2]
    all_dx_dt.append(dx_dt)
    all_x.append(x_run)
all_x = all_x[:1] # take the first 1 run only
all_dx_dt = all_dx_dt[:1] # take the first 1 run only
dx_dt = torch.cat(all_dx_dt, dim=0)
x_data = torch.cat(all_x, dim=0)


train_x_data = x_data[:, :, :]
train_dx_dt = dx_dt[:, :, :]
adj_matrix_tensor = torch.tensor(adj_matrix.values, dtype=torch.float32).to(runtimeconfig.device)
x_data_tensor_ds = TensorDataset(train_x_data, train_dx_dt)
if runtimeconfig.device == "cuda":
    dataloader = DataLoader(x_data_tensor_ds, batch_size=512, shuffle=True, pin_memory=True)
else:
    dataloader = DataLoader(x_data_tensor_ds, batch_size=512, shuffle=True)
# dataloader = DataLoader(x_data_tensor_ds, batch_size=1024, shuffle=True)

controller_config = ControllerConfig(
    input_dim=20,
    hidden_dim=64,
    lr=0.001,
    num_epochs=10,
    num_cands_per_epoch=10,
    percentile_threshold=0.4,
    num_trees=2
)
fex_config = FEXConfig(
    num_epochs=1000,
    bfgs_epochs=0,
    bfgs_lr=0.1,
    leaf_dim=x_data.shape[2],
    num_leaves=forcing_tree_config.num_leaves,
    weight_decay=0.0,
    mag_entropy_weight=0.6
)
best_candidates = train_network_controller(
    forcing_tree_config,
    inter_tree_config,
    dataloader,
    adj_matrix_tensor,
    controller_config,
    fex_config,
    num_processes=1
)
best_candidates.save_candidates(str(run_dir / "pre_finetune/best_candidates.pt"))
best_candidates.visualize_candidates(str(run_dir / "pre_finetune/candidate_viz"), clear_directory=True)