import argparse
import math
import logging
import logging
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

def setup_run_dir(num_epochs, batchsize, job_id=None, mode="default") -> Path:
    if job_id:
        job_id = str(job_id)
    else:
        job_id = ""
    run_id = f"e_{num_epochs}_b_{batchsize}_{job_id}"
    run_dir = HR_DIR / "logs" / f"{mode}" / f"run_{run_id}"
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
    resolution_factor = 1
    resolution_factor = 1

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




def independent_t_test_greater(sample_a: np.ndarray, sample_b: np.ndarray) -> tuple[float, float]:

    result = ttest_ind(sample_a, sample_b, equal_var=True)
    t_stat = float(result.statistic)
    two_sided_p = float(result.pvalue)

    one_sided_p = two_sided_p / 2.0 if t_stat >= 0 else 1.0 - (two_sided_p / 2.0)
    return t_stat, one_sided_p


def _log_fex_trees(
    forcing_op_indices: list,
    inter_op_indices: list,
    fex_config: "FEXConfig",
    save_dir: Path,
    label: str,
    logger,
):
    """Log the text expression of both trees and save graphviz PNGs to save_dir/visualizations/."""
    from FEX.training.tree_helpers import visualize_tree
    forcing_tree_config = get_tree_config("depth_3_leaves_4_config")
    inter_tree_config = get_tree_config("depth_2_tree_config")
    device = runtimeconfig.device

    f_idx = torch.tensor(forcing_op_indices, dtype=torch.long).to(device)
    i_idx = torch.tensor(inter_op_indices, dtype=torch.long).to(device)

    forcing_fex = FEX(
        sample_indices=f_idx,
        leaf_dim=fex_config.leaf_dim,
        num_leaves=forcing_tree_config.num_leaves,
        tree_structure=forcing_tree_config,
        init_tau=fex_config.tau_start,
    ).to(device)
    inter_fex = FEX(
        sample_indices=i_idx,
        leaf_dim=fex_config.leaf_dim * 2,
        num_leaves=inter_tree_config.num_leaves,
        tree_structure=inter_tree_config,
        init_tau=fex_config.tau_start,
    ).to(device)
    _apply_inter_leaf_masks(inter_fex, fex_config.leaf_dim)

    logger.info(f"[{label}] forcing tree: {str(forcing_fex)}")
    logger.info(f"[{label}] inter tree:   {str(inter_fex)}")

    viz_dir = save_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    safe_label = label.replace(" ", "_").lower()
    try:
        visualize_tree(forcing_fex, filename=str(viz_dir / f"{safe_label}_forcing_tree"))
        visualize_tree(inter_fex, filename=str(viz_dir / f"{safe_label}_inter_tree"))
    except Exception as e:
        logger.warning(f"[{label}] graphviz render failed: {e}")


def _apply_inter_leaf_masks(inter_fex: FEX, node_dim: int) -> None:
    """Match train_fex masking: leaf0 uses self dims, leaf1 uses neighbor dims."""
    if len(inter_fex.leaf_mlps) < 2:
        return
    with torch.no_grad():
        if hasattr(inter_fex.leaf_mlps[0], "logit_mask"):
            inter_fex.leaf_mlps[0].logit_mask[node_dim:].fill_(-1e9)
            inter_fex.leaf_mlps[1].logit_mask[:node_dim].fill_(-1e9)


def _save_top_candidate_visualizations(
    top_records: list[dict],
    save_dir: Path,
    label: str,
    logger,
) -> None:
    """Save per-sample top candidate trees and a short summary table."""
    from FEX.training.tree_helpers import visualize_tree

    logger = logger or logging.getLogger(__name__)

    if not top_records:
        logger.warning(f"[{label}] no sample candidates available for visualization")
        return

    safe_label = label.replace(" ", "_").lower()
    top_dir = save_dir / "top_candidates" / safe_label
    top_dir.mkdir(parents=True, exist_ok=True)

    lines = ["rank,sample_idx,reward"]
    for rank, rec in enumerate(top_records, start=1):
        reward = float(rec["reward"])
        sample_idx = int(rec["sample_idx"])
        lines.append(f"{rank},{sample_idx},{reward:.8f}")

        prefix = f"rank_{rank:02d}_sample_{sample_idx:03d}_reward_{reward:.6f}"
        try:
            visualize_tree(rec["forcing_tree"], filename=str(top_dir / f"{prefix}_forcing_tree"))
            visualize_tree(rec["inter_tree"], filename=str(top_dir / f"{prefix}_inter_tree"))
        except Exception as e:
            logger.warning(f"[{label}] failed to render top candidate rank {rank}: {e}")

    (top_dir / "summary.csv").write_text("\n".join(lines) + "\n")
    logger.info(f"[{label}] saved top candidate visualizations to {top_dir}")


def _run_fex_samples(
    result_queue: mp.Queue,
    forcing_op_indices: list,
    inter_op_indices: list,
    num_samples: int,
    fex_config: "FEXConfig",
    x_data: torch.Tensor,
    dx_dt: torch.Tensor,
    adj_matrix_tensor: torch.Tensor,
    batch_size: int,
    save_dir: Path,
    label: str,
    top_k: int,
):

    forcing_tree_config = get_tree_config("depth_3_leaves_4_config")
    inter_tree_config = get_tree_config("depth_2_tree_config")

    dataset = TensorDataset(x_data, dx_dt)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = runtimeconfig.device
    f_indices = torch.tensor(forcing_op_indices, dtype=torch.long).to(device)
    i_indices = torch.tensor(inter_op_indices, dtype=torch.long).to(device)

    forcing_fex = FEX(
        sample_indices=f_indices,
        leaf_dim=fex_config.leaf_dim,
        num_leaves=forcing_tree_config.num_leaves,
        tree_structure=forcing_tree_config,
        init_tau=fex_config.tau_start,
    ).to(device)
    inter_fex = FEX(
        sample_indices=i_indices,
        leaf_dim=fex_config.leaf_dim * 2,
        num_leaves=inter_tree_config.num_leaves,
        tree_structure=inter_tree_config,
        init_tau=fex_config.tau_start,
    ).to(device)
    _apply_inter_leaf_masks(inter_fex, fex_config.leaf_dim)

    rewards = []
    top_records: list[dict] = []
    for sample_idx in range(num_samples):
        score = train_network_fex(
            forcing_fex, inter_fex, dataloader, adj_matrix_tensor, fex_config, verbose=False,
        )
        reward = 1.0 / (1.0 + score)
        rewards.append(reward)

        rec = {
            "reward": float(reward),
            "sample_idx": int(sample_idx),
            "forcing_tree": forcing_fex.copy_inorder().cpu(),
            "inter_tree": inter_fex.copy_inorder().cpu(),
        }
        top_records.append(rec)
        top_records.sort(key=lambda r: r["reward"], reverse=True)
        if len(top_records) > max(1, top_k):
            top_records.pop(-1)

        forcing_fex.reset(fex_config)
        inter_fex.reset(fex_config)
        _apply_inter_leaf_masks(inter_fex, fex_config.leaf_dim)

    _save_top_candidate_visualizations(top_records, save_dir=save_dir, label=label, logger=runtimeconfig.train_logger)

    result_queue.put(rewards)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--job_id", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=30)
    parser.add_argument("--top_k_viz", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=60)
    parser.add_argument("--mode", type=str, default="default", choices=["default", "sigmoid"])
    args = parser.parse_args()

    run_dir = setup_run_dir(args.num_epochs, args.batch_size, job_id=args.job_id, mode=args.mode)

    log_path = run_dir / "controller_eval.log"
    runtimeconfig.CreateLogger(str(log_path), name="train_logger")
    train_logger = runtimeconfig.train_logger
    train_logger.info(
        f"Starting eval_fex: mode={args.mode}, job_id={args.job_id}, "
        f"batch_size={args.batch_size}, num_epochs={args.num_epochs}, num_samples={args.num_samples}"
    )

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
        pct_cosine_restart=1.0,
        tau_start=4.0,
        tau_end=0.015,
    )

    # Ground-truth indices (immutable baseline for comparison)
    ground_truth_forcing_op_indices = torch.tensor([0, 0, 0, 0, 1, 2, 0], dtype=torch.long).to(runtimeconfig.device)
    ground_truth_inter_op_indices = torch.tensor([1, 0, 3], dtype=torch.long).to(runtimeconfig.device)
    
    save_dir = run_dir / "pre_finetune"
    random_init_rewards = []
    random_ground_truth_rewards = []

    if args.mode == "default":
        train_logger.info("Mode default: evaluating random controller-sampled candidates")
        _log_fex_trees(
            ground_truth_forcing_op_indices.tolist(),
            ground_truth_inter_op_indices.tolist(),
            fex_config, save_dir, "ground_truth", train_logger,
        )
        t1 = time.time()
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
        t2 = time.time()
        approx_time_per_cand = (t2 - t1) / args.num_samples
        # Ground truth runs sequentially (controller already consumed workers)
        train_logger.info("Evaluating ground-truth baseline (sequential)")
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
        _apply_inter_leaf_masks(inter_fex, fex_config.leaf_dim)
        t1 = time.time()
        gt_top_records: list[dict] = []
        for sample_idx in range(args.num_samples):
            score = train_network_fex(
                forcing_fex, inter_fex, dataloader, adj_matrix_tensor, fex_config,
                use_entropy=False, verbose=(sample_idx == 0),
            )
            reward = 1.0 / (1.0 + score)
            random_ground_truth_rewards.append(reward)

            rec = {
                "reward": float(reward),
                "sample_idx": int(sample_idx),
                "forcing_tree": forcing_fex.copy_inorder().cpu(),
                "inter_tree": inter_fex.copy_inorder().cpu(),
            }
            gt_top_records.append(rec)
            gt_top_records.sort(key=lambda r: r["reward"], reverse=True)
            if len(gt_top_records) > max(1, args.top_k_viz):
                gt_top_records.pop(-1)

            forcing_fex.reset(fex_config)
            inter_fex.reset(fex_config)
            _apply_inter_leaf_masks(inter_fex, fex_config.leaf_dim)
        t2 = time.time()
        approx_rate_single_core = (t2 - t1) / args.num_samples

        _save_top_candidate_visualizations(
            gt_top_records,
            save_dir=save_dir,
            label="ground_truth",
            logger=train_logger,
        )

        controller_top_records = []
        for rank, cand in enumerate(best_candidates, start=1):
            if rank > max(1, args.top_k_viz):
                break
            controller_top_records.append(
                {
                    "reward": float(cand.reward),
                    "sample_idx": int(cand.id),
                    "forcing_tree": cand.forcing_tree,
                    "inter_tree": cand.inter_tree,
                }
            )
        _save_top_candidate_visualizations(
            controller_top_records,
            save_dir=save_dir,
            label="test",
            logger=train_logger,
        )

    else:
        # sigmoid / inter_test 
        if args.mode == "sigmoid":
            train_logger.info("Mode sigmoid: baseline uses inter tree mul(identity, sigmoid)")
            candidate_forcing_op_indices = [0, 0, 0, 0, 3, 3, 3]
            candidate_inter_op_indices = [1, 0, 3]
        else:  # inter_test
            train_logger.info("Mode inter_test: baseline uses inter tree mul(sigmoid, sigmoid)")
            candidate_forcing_op_indices = [0, 0, 0, 0, 1, 2, 0]
            candidate_inter_op_indices = [1, 3, 3]

        _log_fex_trees(
            candidate_forcing_op_indices,
            candidate_inter_op_indices,
            fex_config, save_dir, args.mode, train_logger,
        )
        _log_fex_trees(
            ground_truth_forcing_op_indices.tolist(),
            ground_truth_inter_op_indices.tolist(),
            fex_config, save_dir, "ground_truth", train_logger,
        )
        train_logger.info("Launching test and ground-truth FEX evaluations in parallel (2 cores)")
        result_q_test = mp.Queue()
        result_q_gt = mp.Queue()
        p_test = mp.Process(
            target=_run_fex_samples,
            args=(
                result_q_test,
                candidate_forcing_op_indices,
                candidate_inter_op_indices,
                args.num_samples,
                fex_config,
                x_data,
                dx_dt,
                adj_matrix_tensor,
                args.batch_size,
                save_dir,
                "test",
                args.top_k_viz,
            ),
        )
        p_gt = mp.Process(
            target=_run_fex_samples,
            args=(
                result_q_gt,
                ground_truth_forcing_op_indices.tolist(),
                ground_truth_inter_op_indices.tolist(),
                args.num_samples,
                fex_config,
                x_data,
                dx_dt,
                adj_matrix_tensor,
                args.batch_size,
                save_dir,
                "ground_truth",
                args.top_k_viz,
            ),
        )
        t1 = time.time()
        p_test.start()
        p_gt.start()
        p_test.join()
        p_gt.join()
        t2 = time.time()
        if p_test.exitcode != 0:
            raise RuntimeError(f"Test FEX worker exited with code {p_test.exitcode}")
        if p_gt.exitcode != 0:
            raise RuntimeError(f"Ground-truth FEX worker exited with code {p_gt.exitcode}")
        random_init_rewards = result_q_test.get()
        random_ground_truth_rewards = result_q_gt.get()
        random_init_rewards.sort(reverse=True)
        approx_time_per_cand = (t2 - t1) / args.num_samples
        approx_rate_single_core = approx_time_per_cand  # both ran simultaneously
    random_init_rewards_np = np.asarray(random_init_rewards, dtype=np.float64)
    random_ground_truth_rewards_np = np.asarray(random_ground_truth_rewards, dtype=np.float64)
    t_stat, p_value = independent_t_test_greater(random_ground_truth_rewards_np, random_init_rewards_np)

    conclusion = (
        "Reject H0: ground-truth mean reward is higher than random-sequence mean reward"
        if p_value < args.alpha
        else "Fail to reject H0"
    )
    baseline_name_map = {
        "default": "random_sequence",
        "sigmoid": "sigmoid_sequence",
        "inter_test": "sigmoid_mul_sigmoid_sequence",
    }
    baseline_name = baseline_name_map[args.mode]
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
        train_logger.info(line)
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
    name_map = {
        "default": "Random",
        "sigmoid": "Sigmoid",
        "inter_test": "Sigmoid-Mul-Sigmoid",
    }
    name = name_map[args.mode]
    plt.legend([f"Average Reward ({name} Sequences): {np.mean(random_init_rewards):.4f}", f"Average Reward (Ground Truth): {np.mean(random_ground_truth_rewards):.4f}"])
    plt.figtext(0.02, 0.08, f"Independent t-test (one-sided) p={p_value:.3g}", ha="left", fontsize=9)
    plt.figtext(0.02, 0.12, f"time per cand (2 cores): {approx_time_per_cand:.2f} seconds", ha="left", fontsize=9)
    plt.figtext(0.02, 0.16, f"approx time per cand (single core): {approx_rate_single_core:.2f} seconds", ha="left", fontsize=9)
    plt.savefig(save_dir / "random_sample_rewards.png")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()