import argparse
import math
import os
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from FEX.models.learnable_tree import FEX
from FEX.training.train_fex import train_network_fex
import torch
from torch.utils.data import DataLoader, TensorDataset

from FEX.training.train_configs import FEXConfig, runtimeconfig
from FEX.training.train_controller import ControllerConfig, train_network_controller
from FEX.utils.numerical_deriv import NumericalDeriv
from FEX.utils.tree_configs import get_tree_config
from FEX.utils.operations import UNARY_OPS, BINARY_OPS


SCRIPT_DIR = Path(__file__).resolve().parent
HR_DIR = SCRIPT_DIR.parent
DATA_DIR = HR_DIR / "data"

def setup_run_dir(job_id=None) -> Path:
    if job_id:
        job_id = str(job_id)
    else:
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
    resolution_factor = 4

    num_runs = num_timesteps // per_run_timesteps
    x_chunks = torch.chunk(x_data, num_runs, dim=0)

    all_x = []
    all_dx_dt = []

    for x_run in x_chunks:
        x_run = x_run[::resolution_factor]
        new_dt = dt * resolution_factor
        dx_dt = NumericalDeriv(x_run, dt=new_dt)
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


def make_dataloader(x_data, dx_dt, batch_size=256):
    dataset = TensorDataset(x_data, dx_dt)

    pin_memory = torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
    )


def validate_ground_truth_operators(forcing_op_indices, inter_op_indices):
    """
    Validate that operator indices are correct for ground truth specification.
    Ground truth:
    - Forcing: all binary=add, unary={identity, square, cube, identity}
    - Inter: binary=mul, unary={identity, sigmoid}
    """
    errors = []
    
    # Check forcing tree indices
    for i, idx in enumerate(forcing_op_indices[:3]):
        if idx >= len(BINARY_OPS):
            errors.append(f"Forcing binary op index {i}: {idx} out of range [0, {len(BINARY_OPS)-1}]")
        elif BINARY_OPS[idx].__name__ != "add":
            errors.append(f"Forcing binary op {i}: expected 'add', got '{BINARY_OPS[idx].__name__}'")
    
    for i, idx in enumerate(forcing_op_indices[3:], start=3):
        if idx >= len(UNARY_OPS):
            errors.append(f"Forcing unary op index {i}: {idx} out of range [0, {len(UNARY_OPS)-1}]")
        elif UNARY_OPS[idx].__name__ not in ["identity", "square", "cube"]:
            errors.append(f"Forcing unary op {i}: expected one of {{identity, square, cube}}, got '{UNARY_OPS[idx].__name__}'")
    
    # Check inter tree indices
    if inter_op_indices[0] >= len(BINARY_OPS):
        errors.append(f"Inter binary op: {inter_op_indices[0]} out of range [0, {len(BINARY_OPS)-1}]")
    elif BINARY_OPS[inter_op_indices[0]].__name__ != "mul":
        errors.append(f"Inter binary op: expected 'mul', got '{BINARY_OPS[inter_op_indices[0]].__name__}'")
    
    if inter_op_indices[1] >= len(UNARY_OPS):
        errors.append(f"Inter unary op 1: {inter_op_indices[1]} out of range [0, {len(UNARY_OPS)-1}]")
    elif UNARY_OPS[inter_op_indices[1]].__name__ != "identity":
        errors.append(f"Inter unary op 1: expected 'identity', got '{UNARY_OPS[inter_op_indices[1]].__name__}'")
    
    if inter_op_indices[2] >= len(UNARY_OPS):
        errors.append(f"Inter unary op 2: {inter_op_indices[2]} out of range [0, {len(UNARY_OPS)-1}]")
    elif UNARY_OPS[inter_op_indices[2]].__name__ != "sigmoid":
        errors.append(f"Inter unary op 2: expected 'sigmoid', got '{UNARY_OPS[inter_op_indices[2]].__name__}'")
    
    if errors:
        error_msg = "Ground truth operator validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValueError(error_msg)
    
    print("Ground truth operators validated successfully")
    print(f"  Forcing: binary={BINARY_OPS[forcing_op_indices[0]].__name__}, unary=[{', '.join(UNARY_OPS[i].__name__ for i in forcing_op_indices[3:])}]")
    print(f"  Inter: binary={BINARY_OPS[inter_op_indices[0]].__name__}, unary=[{UNARY_OPS[inter_op_indices[1]].__name__}, {UNARY_OPS[inter_op_indices[2]].__name__}]")


def independent_t_test_greater(sample_a: np.ndarray, sample_b: np.ndarray) -> tuple[float, float]:

    result = ttest_ind(sample_a, sample_b, equal_var=True)
    t_stat = float(result.statistic)
    two_sided_p = float(result.pvalue)

    one_sided_p = two_sided_p / 2.0 if t_stat >= 0 else 1.0 - (two_sided_p / 2.0)
    return t_stat, one_sided_p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--job_id", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--num_epochs", type=int, default=60)
    parser.add_argument("--mode", type=str, default="default", choices=["default", "sigmoid"])
    args = parser.parse_args()

    run_dir = setup_run_dir(job_id=args.job_id)

    log_path = run_dir / "controller_eval.log"
    runtimeconfig.CreateLogger(str(log_path), name="train_logger")

    forcing_tree_config = get_tree_config("depth_3_leaves_4_config")
    inter_tree_config = get_tree_config("depth_2_tree_config")

    x_data, dx_dt, adj_matrix_tensor = load_hr_data()
    dataloader = make_dataloader(x_data, dx_dt, args.batch_size)

    controller_config = ControllerConfig(
        input_dim=20,
        hidden_dim=64,
        lr=0.001,
        num_epochs=1,
        num_cands_per_epoch=args.num_samples,
        percentile_threshold=1.0, # catch all candidates
        num_trees=2,
        poolsize=50,
        epsilon_greedy=1.0 # full random sample
    )

    fex_config = FEXConfig(
        num_epochs=args.num_epochs,
        bfgs_epochs=0,
        lr=0.15,
        inter_lr=0.05,
        bfgs_lr=0.1,
        leaf_dim=x_data.shape[2],
        num_leaves=forcing_tree_config.num_leaves,
        mag_entropy_weight=0.0,
        pct_cosine_restart=1.0,
        tau_start=4.0,
        tau_end=0.015,
    )

    # Ground-truth indices (immutable baseline for comparison)
    ground_truth_forcing_op_indices = torch.tensor([0, 0, 0, 0, 1, 2, 0], dtype=torch.long).to(runtimeconfig.device)
    ground_truth_inter_op_indices = torch.tensor([1, 0, 3], dtype=torch.long).to(runtimeconfig.device)
    validate_ground_truth_operators(
        forcing_op_indices=ground_truth_forcing_op_indices,
        inter_op_indices=ground_truth_inter_op_indices,
    )
    
    save_dir = run_dir / "pre_finetune"
    random_init_rewards = []
    t1 = time.time()
    if args.mode=="default":
        best_candidates = train_network_controller(
            forcing_tree_config,
            inter_tree_config,
            dataloader,
            adj_matrix_tensor,
            controller_config,
            fex_config,
            checkpoint_dir=None,
            num_workers=args.num_workers,
        )
        random_init_rewards = [cand.reward for cand in best_candidates]
    # end of default mode block
    elif args.mode=="sigmoid": # eval random initializations for sequence {identity, sigmoid, sigmoid, sigmoid}
        candidate_forcing_op_indices = torch.tensor([0, 0, 0, 0, 3, 3, 3], dtype=torch.long).to(runtimeconfig.device)
        candidate_inter_op_indices = torch.tensor([1, 0, 3], dtype=torch.long).to(runtimeconfig.device)
        forcing_fex = FEX(
            sample_indices=candidate_forcing_op_indices,
            leaf_dim=fex_config.leaf_dim,
            num_leaves=forcing_tree_config.num_leaves,
            tree_structure=forcing_tree_config,
            init_tau=fex_config.tau_start,
        ).to(runtimeconfig.device)

        inter_fex = FEX(
            sample_indices=candidate_inter_op_indices,
            leaf_dim=fex_config.leaf_dim * 2,
            num_leaves=inter_tree_config.num_leaves,
            tree_structure=inter_tree_config,
            init_tau=fex_config.tau_start,
        ).to(runtimeconfig.device)
        for _ in range(args.num_samples):
            score = train_network_fex(forcing_fex, inter_fex, dataloader, adj_matrix_tensor, fex_config, use_entropy=False, verbose=False)
            reward = 1.0 / (1.0 + score)
            random_init_rewards.append(reward)
            forcing_fex.reset(fex_config)
            inter_fex.reset(fex_config)
        random_init_rewards.sort(reverse=True)
    # end of sigmoid mode block
    t2 = time.time()
    approx_time_per_cand = (t2 - t1) / args.num_samples

    # sample of random initializations for true operator sequence
    # -----------------------------------------------------------

    random_ground_truth_rewards = []    
    forcing_fex = FEX(
        sample_indices=ground_truth_forcing_op_indices,
        leaf_dim=fex_config.leaf_dim,
        num_leaves=forcing_tree_config.num_leaves,
        tree_structure=forcing_tree_config,
        init_tau=fex_config.tau_start,
    ).to(runtimeconfig.device)

    inter_fex = FEX(
        sample_indices=ground_truth_inter_op_indices,
        leaf_dim=fex_config.leaf_dim * 2,
        num_leaves=inter_tree_config.num_leaves,
        tree_structure=inter_tree_config,
        init_tau=fex_config.tau_start,
    ).to(runtimeconfig.device)

    t1 = time.time()
    for _ in range(args.num_samples):
        score = train_network_fex(forcing_fex, inter_fex, dataloader, adj_matrix_tensor, fex_config, use_entropy=False, verbose=False)
        reward = 1.0 / (1.0 + score)
        random_ground_truth_rewards.append(reward)
        forcing_fex.reset(fex_config)
        inter_fex.reset(fex_config)
    t2 = time.time()
    approx_rate_single_core = (t2 - t1) / args.num_samples
    random_init_rewards_np = np.asarray(random_init_rewards, dtype=np.float64)
    random_ground_truth_rewards_np = np.asarray(random_ground_truth_rewards, dtype=np.float64)
    t_stat, p_value = independent_t_test_greater(random_ground_truth_rewards_np, random_init_rewards_np)

    conclusion = (
        "Reject H0: ground-truth mean reward is higher than random-sequence mean reward"
        if p_value < args.alpha
        else "Fail to reject H0"
    )
    baseline_name = "random_sequence" if args.mode == "default" else "sigmoid_sequence"
    test_lines = [
        "One-sided independent two-sample t-test",
        f"H0: mean(ground_truth_rewards) <= mean({baseline_name}_rewards)",
        f"H1: mean(ground_truth_rewards) > mean({baseline_name}_rewards)",
        f"mean_ground_truth: {np.mean(random_ground_truth_rewards_np):.6f}",
        f"mean_{baseline_name}: {np.mean(random_init_rewards_np):.6f}",
        f"t_statistic: {t_stat:.6f}",
        f"one_sided_p_value: {p_value:.6g}",
        f"conclusion_alpha_{args.alpha}: {conclusion}",
    ]
    for line in test_lines:
        print(line)
    (save_dir / "hypothesis_test.txt").write_text("\n".join(test_lines) + "\n")

    random_ground_truth_rewards.sort(reverse=True)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.subplots_adjust(bottom=0.20)
    plt.plot(random_init_rewards)
    plt.plot(random_ground_truth_rewards)
    plt.xlabel("Candidate Index")
    plt.ylabel("Reward")
    plt.title("Rewards of Candidate Random Sample")
    name = "Random" if args.mode=="default" else "Sigmoid"
    plt.legend([f"Average Reward ({name} Sequences): {np.mean(random_init_rewards):.4f}", f"Average Reward (Ground Truth): {np.mean(random_ground_truth_rewards):.4f}"])
    plt.figtext(0.02, 0.08, f"Independent t-test (one-sided) p={p_value:.3g}", ha="left", fontsize=9)
    plt.figtext(0.02, 0.12, f"time per cand (2 cores): {approx_time_per_cand:.2f} seconds", ha="left", fontsize=9)
    plt.figtext(0.02, 0.16, f"approx time per cand (single core): {approx_rate_single_core:.2f} seconds", ha="left", fontsize=9)
    plt.savefig(save_dir / "random_sample_rewards.png")


if __name__ == "__main__":
    main()
