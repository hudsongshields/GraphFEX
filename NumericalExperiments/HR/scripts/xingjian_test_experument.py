"""HR diagnostics for RBM interaction information loss.

This script tests the failure mode described in the xingjian_check notes:
Random Batch Method (RBM) keeps only within-batch edges during candidate
scoring.  The interaction estimator without inverse-probability correction is
shrunk by roughly p / N, and low-degree nodes quickly lose all interaction
coverage as N grows.

The experiment avoids dense N x N adjacency matrices.  It samples a sparse
directed scale-free edge list, evaluates the true HR interaction term

    H_i = 0.15 * (2 - x_i) * sum_j A_ij sigmoid(x_j),

and compares it with unweighted and reweighted RBM estimates.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from FEX.helpers.tree_configs import get_tree_config
from FEX.models.learnable_tree import FEX


COUPLING_STRENGTH = 0.15
EPS = 1e-12
TRUE_SELF_COEFF = torch.tensor([3.24, 0.0, 3.0, -1.0, 1.0, -1.0], dtype=torch.float64)


@dataclass
class ScalingRow:
    nodes: int
    edges: int
    target_mean_degree: float
    rbm_batch_size: int
    trial: int
    coverage_fraction: float
    mean_retained_degree: float
    retained_edges: int
    smape_unweighted: float
    smape_reweighted: float
    fit_scale_unweighted: float
    fit_scale_partition_average: float
    fit_scale_reweighted: float
    expected_inverse_keep_scale: float


@dataclass
class DegreeScoreRow:
    subset: str
    candidate: str
    nodes: int
    edges: int
    fitted_scale: float
    mse: float
    smape: float


@dataclass
class FEXStratifiedRow:
    nodes: int
    mode: str
    candidate: str
    score_rank: int
    score_nodes: int
    score_edges: int
    pair_evaluations: int
    full_edges: int
    coverage_fraction: float
    score_mse: float
    score_smape: float
    full_mse: float
    full_smape: float
    elapsed_seconds: float


@dataclass
class JointCandidateRow:
    nodes: int
    snr_db: str
    mode: str
    candidate: str
    score_rank: int
    score_nodes: int
    score_edges: int
    pair_evaluations: int
    full_edges: int
    coverage_fraction: float
    dx_mse: float
    score_interaction_smape: float
    full_interaction_smape: float
    fitted_coupling: float
    coupling_ratio: float
    self_coeff_l2: float
    elapsed_seconds: float


FEX_CANDIDATES = {
    # depth_2_tree_config sample indices are [binary, unary_left, unary_right].
    # BINARY_OPS: add=0, mul=1, sub=2.
    # UNARY_OPS: identity=0, square=1, cube=2, fourth=3, exp=4, sigmoid=5, sin=6.
    "true_sigmoid_product": [1, 0, 5],
    "product_linear": [1, 0, 0],
    "additive_sigmoid": [0, 0, 5],
    "product_square": [1, 0, 1],
}


def parse_node_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_snr_list(raw: str) -> list[float]:
    values: list[float] = []
    for item in raw.split(","):
        text = item.strip().lower()
        if not text:
            continue
        if text in {"inf", "infty", "infinite", "none"}:
            values.append(math.inf)
        else:
            values.append(float(text))
    return values


def format_snr(value: float) -> str:
    if math.isinf(value):
        return "inf"
    return f"{value:g}"


def smape(pred: torch.Tensor, target: torch.Tensor) -> float:
    value = 2.0 * (pred - target).abs() / (pred.abs() + target.abs() + EPS)
    return float(100.0 * value.mean().item())


def fit_scale(pred_base: torch.Tensor, target: torch.Tensor) -> float:
    denom = torch.sum(pred_base * pred_base)
    if float(denom.item()) <= EPS:
        return float("nan")
    return float((torch.sum(pred_base * target) / denom).item())


def sample_static_scale_free_edges(
    num_nodes: int,
    num_edges: int,
    rng: np.random.Generator,
    *,
    gamma_in: float = 5.0,
    gamma_out: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample directed target/source edges without creating a dense matrix."""
    alpha_in = 1.0 / (gamma_in - 1.0)
    alpha_out = 1.0 / (gamma_out - 1.0)

    idx = 1.0 + np.arange(num_nodes)
    w_in = rng.permutation(1.0 / (idx**alpha_in))
    w_out = rng.permutation(1.0 / (idx**alpha_out))
    p_in = w_in / np.sum(w_in)
    p_out = w_out / np.sum(w_out)

    unique_edges: set[tuple[int, int]] = set()
    sample_count = max(4 * num_edges, 1024)
    while len(unique_edges) < num_edges:
        sampled_sources = rng.choice(num_nodes, size=sample_count, p=p_out)
        sampled_targets = rng.choice(num_nodes, size=sample_count, p=p_in)
        for source, target in zip(sampled_sources, sampled_targets):
            if source != target:
                unique_edges.add((int(target), int(source)))
                if len(unique_edges) >= num_edges:
                    break
        sample_count = int(sample_count * 1.5)

    edge_array = np.array(sorted(unique_edges)[:num_edges], dtype=np.int64)
    dst = torch.from_numpy(edge_array[:, 0]).long()
    src = torch.from_numpy(edge_array[:, 1]).long()
    weight = torch.ones(num_edges, dtype=torch.float32)
    return dst, src, weight


def sample_hr_x(num_samples: int, num_nodes: int, generator: torch.Generator) -> torch.Tensor:
    x = torch.empty(num_samples, num_nodes, dtype=torch.float32)
    x.uniform_(-2.0, 2.0, generator=generator)
    return x


def sample_hr_states(num_samples: int, num_nodes: int, generator: torch.Generator) -> torch.Tensor:
    states = torch.empty(num_samples, num_nodes, 3, dtype=torch.float32)
    states[:, :, 0].uniform_(-2.0, 2.0, generator=generator)
    states[:, :, 1].uniform_(-8.0, 4.0, generator=generator)
    states[:, :, 2].uniform_(0.0, 5.0, generator=generator)
    return states


def aggregate_hr_interaction(
    x: torch.Tensor,
    dst: torch.Tensor,
    src: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    num_samples, num_nodes = x.shape
    messages = torch.sigmoid(x[:, src]) * weight.view(1, -1)
    neighbor_sum = torch.zeros(num_samples, num_nodes, dtype=x.dtype)
    neighbor_sum.scatter_add_(1, dst.view(1, -1).expand(num_samples, -1), messages)
    return COUPLING_STRENGTH * (2.0 - x) * neighbor_sum


def aggregate_hr_interaction_from_states(
    states: torch.Tensor,
    dst: torch.Tensor,
    src: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return aggregate_hr_interaction(states[:, :, 0], dst, src, weight)


def sparse_hr_rhs(
    state: torch.Tensor,
    dst: torch.Tensor,
    src: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    x = state[:, 0]
    y = state[:, 1]
    z = state[:, 2]

    messages = torch.sigmoid(x[src]) * weight
    neighbor_sum = torch.zeros_like(x)
    neighbor_sum.scatter_add_(0, dst, messages)
    interaction = COUPLING_STRENGTH * (2.0 - x) * neighbor_sum

    dx = y - x.pow(3) + 3.0 * x.pow(2) - z + 3.24 + interaction
    dy = 1.0 - 5.0 * x.pow(2) - y
    dz = 0.005 * (4.0 * (x + 1.6) - z)
    return torch.stack((dx, dy, dz), dim=-1)


def sparse_hr_rk4_step(
    state: torch.Tensor,
    dst: torch.Tensor,
    src: torch.Tensor,
    weight: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    k1 = sparse_hr_rhs(state, dst, src, weight)
    k2 = sparse_hr_rhs(state + 0.5 * dt * k1, dst, src, weight)
    k3 = sparse_hr_rhs(state + 0.5 * dt * k2, dst, src, weight)
    k4 = sparse_hr_rhs(state + dt * k3, dst, src, weight)
    return state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def add_gaussian_noise_db(
    data: torch.Tensor,
    snr_db: float,
    generator: torch.Generator,
) -> torch.Tensor:
    if math.isinf(snr_db):
        return data.clone()
    column_variances = torch.var(data, dim=0, correction=1)
    signal_power = column_variances.mean()
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = torch.randn(data.shape, generator=generator, dtype=data.dtype) * torch.sqrt(noise_power)
    return data + noise


def five_point_derivative(timeseries: torch.Tensor, dt: float) -> torch.Tensor:
    return (
        timeseries[:-4]
        - 8.0 * timeseries[1:-3]
        + 8.0 * timeseries[3:-1]
        - timeseries[4:]
    ) / (12.0 * dt)


def make_sparse_hr_timeseries(
    num_samples: int,
    num_nodes: int,
    dst: torch.Tensor,
    src: torch.Tensor,
    weight: torch.Tensor,
    *,
    snr_db: float,
    dt: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    states = torch.empty(num_samples, num_nodes, 3, dtype=torch.float32)
    states[0, :, 0].uniform_(-2.0, 2.0, generator=generator)
    states[0, :, 1].uniform_(-8.0, 4.0, generator=generator)
    states[0, :, 2].uniform_(0.0, 5.0, generator=generator)

    with torch.no_grad():
        for step in range(1, num_samples):
            states[step] = sparse_hr_rk4_step(states[step - 1], dst, src, weight, dt)

    observed = add_gaussian_noise_db(states, snr_db, generator)
    derivatives = five_point_derivative(observed, dt)
    return observed[2:-2], derivatives


def rbm_group_ids(
    num_nodes: int,
    batch_size: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    permutation = torch.randperm(num_nodes, generator=generator)
    group_id = torch.empty(num_nodes, dtype=torch.long)
    group_id[permutation] = torch.arange(num_nodes, dtype=torch.long) // batch_size
    group_sizes = torch.bincount(group_id)
    return group_id, group_sizes


def rbm_interaction(
    x: torch.Tensor,
    dst: torch.Tensor,
    src: torch.Tensor,
    weight: torch.Tensor,
    *,
    group_id: torch.Tensor,
    group_sizes: torch.Tensor,
    reweight: bool,
) -> tuple[torch.Tensor, float, float, int]:
    num_samples, num_nodes = x.shape
    keep = group_id[dst] == group_id[src]

    retained_dst = dst[keep]
    retained_src = src[keep]
    retained_weight = weight[keep]

    retained_degree = torch.bincount(retained_dst, minlength=num_nodes)
    coverage = float((retained_degree > 0).float().mean().item())
    mean_retained_degree = float(retained_degree.float().mean().item())
    retained_edges = int(retained_dst.numel())

    if retained_edges == 0:
        return torch.zeros_like(x), coverage, mean_retained_degree, retained_edges

    if reweight:
        sizes = group_sizes[group_id[retained_dst]].float()
        inverse_keep = (num_nodes - 1) / torch.clamp(sizes - 1, min=1.0)
        retained_weight = retained_weight * inverse_keep

    return (
        aggregate_hr_interaction(x, retained_dst, retained_src, retained_weight),
        coverage,
        mean_retained_degree,
        retained_edges,
    )


def run_rbm_scaling(args: argparse.Namespace) -> list[ScalingRow]:
    np_rng = np.random.default_rng(args.seed)
    torch_gen = torch.Generator().manual_seed(args.seed)
    rows: list[ScalingRow] = []

    for num_nodes in args.nodes:
        num_edges = int(round(num_nodes * args.mean_degree))
        dst, src, weight = sample_static_scale_free_edges(
            num_nodes,
            num_edges,
            np_rng,
            gamma_in=args.gamma_in,
            gamma_out=args.gamma_out,
        )

        for trial in range(args.trials):
            x = sample_hr_x(args.state_samples, num_nodes, torch_gen)
            true_interaction = aggregate_hr_interaction(x, dst, src, weight)
            group_id, group_sizes = rbm_group_ids(num_nodes, args.rbm_batch_size, torch_gen)

            unweighted, coverage, mean_retained_degree, retained_edges = rbm_interaction(
                x,
                dst,
                src,
                weight,
                group_id=group_id,
                group_sizes=group_sizes,
                reweight=False,
            )
            reweighted, _, _, _ = rbm_interaction(
                x,
                dst,
                src,
                weight,
                group_id=group_id,
                group_sizes=group_sizes,
                reweight=True,
            )

            averaged_unweighted = torch.zeros_like(unweighted)
            for _ in range(args.scale_partitions):
                avg_group_id, avg_group_sizes = rbm_group_ids(
                    num_nodes,
                    args.rbm_batch_size,
                    torch_gen,
                )
                avg_estimate, _, _, _ = rbm_interaction(
                    x,
                    dst,
                    src,
                    weight,
                    group_id=avg_group_id,
                    group_sizes=avg_group_sizes,
                    reweight=False,
                )
                averaged_unweighted += avg_estimate
            averaged_unweighted /= max(args.scale_partitions, 1)

            scale_unweighted = fit_scale(unweighted, true_interaction)
            scale_partition_average = fit_scale(averaged_unweighted, true_interaction)
            scale_reweighted = fit_scale(reweighted, true_interaction)
            expected_scale = (num_nodes - 1) / max(args.rbm_batch_size - 1, 1)

            rows.append(
                ScalingRow(
                    nodes=num_nodes,
                    edges=num_edges,
                    target_mean_degree=args.mean_degree,
                    rbm_batch_size=args.rbm_batch_size,
                    trial=trial,
                    coverage_fraction=coverage,
                    mean_retained_degree=mean_retained_degree,
                    retained_edges=retained_edges,
                    smape_unweighted=smape(unweighted, true_interaction),
                    smape_reweighted=smape(reweighted, true_interaction),
                    fit_scale_unweighted=scale_unweighted,
                    fit_scale_partition_average=scale_partition_average,
                    fit_scale_reweighted=scale_reweighted,
                    expected_inverse_keep_scale=expected_scale,
                )
            )

    return rows


def initialize_hr_interaction_tree(tree: FEX) -> None:
    """Warm-start a depth-2 FEX on the HR interaction variables.

    This isolates the scorer/stratum effect from random leaf discovery: leaf 0
    starts as 2 - x_i and leaf 1 starts as x_j.  The tree remains trainable.
    """
    if len(tree.leaf_mlps) < 2:
        return

    with torch.no_grad():
        tree.leaf_mlps[0].logits.zero_()
        tree.leaf_mlps[0].logits[0] = -1.0
        tree.leaf_mlps[0].bias.fill_(2.0)

        tree.leaf_mlps[1].logits.zero_()
        tree.leaf_mlps[1].logits[3] = 1.0
        tree.leaf_mlps[1].bias.zero_()

        if tree.parent_node.left is not None and hasattr(tree.parent_node.left.operation, "a"):
            tree.parent_node.left.operation.a.fill_(COUPLING_STRENGTH)
            tree.parent_node.left.operation.b.zero_()
        if tree.parent_node.right is not None and hasattr(tree.parent_node.right.operation, "a"):
            tree.parent_node.right.operation.a.fill_(1.0)
            tree.parent_node.right.operation.b.zero_()


def fex_interaction_prediction(
    tree: FEX,
    states: torch.Tensor,
    dst: torch.Tensor,
    src: torch.Tensor,
    weight: torch.Tensor,
    selected_nodes: torch.Tensor,
) -> torch.Tensor:
    num_samples, num_nodes, _ = states.shape
    if dst.numel() == 0:
        return torch.zeros(num_samples, selected_nodes.numel(), dtype=states.dtype)

    target_states = states[:, dst, :]
    source_states = states[:, src, :]
    edge_input = torch.cat((target_states, source_states), dim=-1)
    edge_values = tree(edge_input.reshape(num_samples * dst.numel(), -1))
    edge_values = edge_values.reshape(num_samples, dst.numel()) * weight.view(1, -1)

    interaction = torch.zeros(num_samples, num_nodes, dtype=states.dtype)
    interaction.scatter_add_(1, dst.view(1, -1).expand(num_samples, -1), edge_values)
    return interaction[:, selected_nodes]


def train_fex_candidate_for_score(
    sample_indices: list[int],
    states: torch.Tensor,
    target: torch.Tensor,
    dst: torch.Tensor,
    src: torch.Tensor,
    weight: torch.Tensor,
    selected_nodes: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[FEX, float, float]:
    tree_config = get_tree_config("depth_2_tree_config")
    best_tree = None
    best_mse = float("inf")
    start = time.perf_counter()

    for restart in range(args.fex_restarts):
        tree = FEX(
            leaf_dim=6,
            sample_indices=sample_indices,
            tree_structure=tree_config,
        )
        if args.fex_hr_warm_start:
            initialize_hr_interaction_tree(tree)

        optimizer = torch.optim.Adam(list(tree.all_parameters()), lr=args.fex_lr)
        for _ in range(args.fex_epochs):
            optimizer.zero_grad()
            prediction = fex_interaction_prediction(
                tree,
                states,
                dst,
                src,
                weight,
                selected_nodes,
            )
            loss = F.mse_loss(prediction, target)
            if not torch.isfinite(loss):
                break
            if float(loss.item()) <= args.fex_stop_mse:
                break
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            prediction = fex_interaction_prediction(
                tree,
                states,
                dst,
                src,
                weight,
                selected_nodes,
            )
            mse = float(F.mse_loss(prediction, target).item())

        if mse < best_mse:
            best_mse = mse
            best_tree = tree

    if best_tree is None:
        raise RuntimeError(f"FEX candidate {sample_indices} produced no finite fit.")

    return best_tree, best_mse, time.perf_counter() - start


def run_fex_stratified_check(args: argparse.Namespace) -> list[FEXStratifiedRow]:
    np_rng = np.random.default_rng(args.seed + 2029)
    torch_gen = torch.Generator().manual_seed(args.seed + 2029)
    rows: list[FEXStratifiedRow] = []

    for num_nodes in args.fex_nodes:
        num_edges = int(round(num_nodes * args.mean_degree))
        dst, src, weight = sample_static_scale_free_edges(
            num_nodes,
            num_edges,
            np_rng,
            gamma_in=args.gamma_in,
            gamma_out=args.gamma_out,
        )
        states = sample_hr_states(args.fex_state_samples, num_nodes, torch_gen)
        true_full = aggregate_hr_interaction_from_states(states, dst, src, weight)
        full_nodes = torch.arange(num_nodes, dtype=torch.long)

        degrees = torch.bincount(dst, minlength=num_nodes)
        all_mask = torch.ones(dst.numel(), dtype=torch.bool)
        requested_modes = set(args.fex_modes)
        score_modes: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, int]] = {}
        if "full" in requested_modes:
            score_modes["full"] = (full_nodes, all_mask, weight, 1.0, int(dst.numel()))

        low_candidates = torch.nonzero(
            (degrees >= 1) & (degrees <= args.fex_low_degree_max),
            as_tuple=True,
        )[0]
        high_threshold = torch.quantile(degrees.float(), 0.9)
        high_candidates = torch.nonzero(degrees.float() >= high_threshold, as_tuple=True)[0]
        low_edges = int(degrees[low_candidates].sum().item())
        high_edges = int(degrees[high_candidates].sum().item())
        stratum_edge_budget = min(args.fex_edge_budget, low_edges, high_edges)

        for mode_name, candidates in (
            ("stratified_low", low_candidates),
            ("top_decile_hubs", high_candidates),
        ):
            selected_nodes = choose_nodes_for_budget(
                candidates,
                degrees,
                stratum_edge_budget,
                torch_gen,
            )
            if selected_nodes.numel() == 0:
                continue
            selected_mask = torch.zeros(num_nodes, dtype=torch.bool)
            selected_mask[selected_nodes] = True
            edge_mask = selected_mask[dst]
            coverage = float((degrees[selected_nodes] > 0).float().mean().item())
            if mode_name in requested_modes:
                score_modes[mode_name] = (
                    selected_nodes,
                    edge_mask,
                    weight[edge_mask],
                    coverage,
                    int(edge_mask.sum().item()),
                )

        if {"rbm_raw", "rbm_reweighted"} & requested_modes:
            group_id, group_sizes = rbm_group_ids(num_nodes, args.rbm_batch_size, torch_gen)
            rbm_mask = group_id[dst] == group_id[src]
            rbm_degree = torch.bincount(dst[rbm_mask], minlength=num_nodes)
            rbm_coverage = float((rbm_degree > 0).float().mean().item())
            rbm_pair_evaluations = int(num_nodes * args.rbm_batch_size)
            if "rbm_raw" in requested_modes:
                score_modes["rbm_raw"] = (
                    full_nodes,
                    rbm_mask,
                    weight[rbm_mask],
                    rbm_coverage,
                    rbm_pair_evaluations,
                )
            if rbm_mask.any():
                sizes = group_sizes[group_id[dst[rbm_mask]]].float()
                rbm_weight = weight[rbm_mask] * (num_nodes - 1) / torch.clamp(sizes - 1, min=1.0)
            else:
                rbm_weight = weight[rbm_mask]
            if "rbm_reweighted" in requested_modes:
                score_modes["rbm_reweighted"] = (
                    full_nodes,
                    rbm_mask,
                    rbm_weight,
                    rbm_coverage,
                    rbm_pair_evaluations,
                )

        for mode_name, (
            selected_nodes,
            edge_mask,
            score_weight,
            coverage,
            pair_evaluations,
        ) in score_modes.items():
            mode_rows: list[FEXStratifiedRow] = []
            score_dst = dst[edge_mask]
            score_src = src[edge_mask]
            score_target = true_full[:, selected_nodes]

            for candidate_name, sample_indices in FEX_CANDIDATES.items():
                tree, score_mse, elapsed = train_fex_candidate_for_score(
                    sample_indices,
                    states,
                    score_target,
                    score_dst,
                    score_src,
                    score_weight,
                    selected_nodes,
                    args,
                )
                with torch.no_grad():
                    score_prediction = fex_interaction_prediction(
                        tree,
                        states,
                        score_dst,
                        score_src,
                        score_weight,
                        selected_nodes,
                    )
                    full_prediction = fex_interaction_prediction(
                        tree,
                        states,
                        dst,
                        src,
                        weight,
                        full_nodes,
                    )

                mode_rows.append(
                    FEXStratifiedRow(
                        nodes=num_nodes,
                        mode=mode_name,
                        candidate=candidate_name,
                        score_rank=0,
                        score_nodes=int(selected_nodes.numel()),
                        score_edges=int(score_dst.numel()),
                        pair_evaluations=pair_evaluations,
                        full_edges=int(dst.numel()),
                        coverage_fraction=coverage,
                        score_mse=score_mse,
                        score_smape=smape(score_prediction, score_target),
                        full_mse=float(F.mse_loss(full_prediction, true_full).item()),
                        full_smape=smape(full_prediction, true_full),
                        elapsed_seconds=elapsed,
                    )
                )

            mode_rows.sort(key=lambda row: row.score_mse)
            for rank, row in enumerate(mode_rows, start=1):
                row.score_rank = rank
            rows.extend(mode_rows)

    return rows


def candidate_base(
    x: torch.Tensor,
    dst: torch.Tensor,
    src: torch.Tensor,
    selected_nodes: torch.Tensor,
    transform_name: str,
) -> torch.Tensor:
    if transform_name == "sigmoid_true":
        feature = torch.sigmoid(x[:, src])
    elif transform_name == "constant_mean":
        feature = torch.full_like(x[:, src], float(torch.sigmoid(x).mean().item()))
    elif transform_name == "affine_sigmoid":
        flat_x = x.reshape(-1)
        flat_y = torch.sigmoid(flat_x)
        slope = torch.mean((flat_x - flat_x.mean()) * (flat_y - flat_y.mean()))
        slope = slope / torch.clamp(torch.var(flat_x, unbiased=False), min=EPS)
        intercept = flat_y.mean() - slope * flat_x.mean()
        feature = intercept + slope * x[:, src]
    elif transform_name == "quadratic":
        centered = x[:, src] - x.mean()
        feature = centered * centered
    else:
        raise ValueError(f"Unknown candidate transform: {transform_name}")

    num_samples, num_nodes = x.shape
    selected_mask = torch.zeros(num_nodes, dtype=torch.bool)
    selected_mask[selected_nodes] = True
    edge_mask = selected_mask[dst]
    local_dst = dst[edge_mask]
    messages = feature[:, edge_mask]

    neighbor_sum = torch.zeros(num_samples, num_nodes, dtype=x.dtype)
    neighbor_sum.scatter_add_(1, local_dst.view(1, -1).expand(num_samples, -1), messages)
    return COUPLING_STRENGTH * (2.0 - x[:, selected_nodes]) * neighbor_sum[:, selected_nodes]


def choose_nodes_for_budget(
    candidates: torch.Tensor,
    degrees: torch.Tensor,
    edge_budget: int,
    generator: torch.Generator,
) -> torch.Tensor:
    if candidates.numel() == 0:
        return candidates
    order = candidates[torch.randperm(candidates.numel(), generator=generator)]
    selected = []
    total_edges = 0
    for node in order.tolist():
        degree = int(degrees[node].item())
        if degree == 0:
            continue
        selected.append(node)
        total_edges += degree
        if total_edges >= edge_budget:
            break
    return torch.tensor(selected, dtype=torch.long)


def build_scoring_modes(
    num_nodes: int,
    dst: torch.Tensor,
    src: torch.Tensor,
    weight: torch.Tensor,
    degrees: torch.Tensor,
    requested_modes: set[str],
    edge_budget: int,
    low_degree_max: int,
    rbm_batch_size: int,
    generator: torch.Generator,
) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, int]]:
    full_nodes = torch.arange(num_nodes, dtype=torch.long)
    score_modes: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, int]] = {}
    all_mask = torch.ones(dst.numel(), dtype=torch.bool)
    if "full" in requested_modes:
        score_modes["full"] = (full_nodes, all_mask, weight, 1.0, int(dst.numel()))

    low_candidates = torch.nonzero(
        (degrees >= 1) & (degrees <= low_degree_max),
        as_tuple=True,
    )[0]
    high_threshold = torch.quantile(degrees.float(), 0.9)
    high_candidates = torch.nonzero(degrees.float() >= high_threshold, as_tuple=True)[0]
    low_edges = int(degrees[low_candidates].sum().item())
    high_edges = int(degrees[high_candidates].sum().item())
    stratum_edge_budget = min(edge_budget, low_edges, high_edges)

    for mode_name, candidates in (
        ("stratified_low", low_candidates),
        ("top_decile_hubs", high_candidates),
    ):
        if mode_name not in requested_modes:
            continue
        selected_nodes = choose_nodes_for_budget(
            candidates,
            degrees,
            stratum_edge_budget,
            generator,
        )
        if selected_nodes.numel() == 0:
            continue
        selected_mask = torch.zeros(num_nodes, dtype=torch.bool)
        selected_mask[selected_nodes] = True
        edge_mask = selected_mask[dst]
        coverage = float((degrees[selected_nodes] > 0).float().mean().item())
        score_modes[mode_name] = (
            selected_nodes,
            edge_mask,
            weight[edge_mask],
            coverage,
            int(edge_mask.sum().item()),
        )

    if {"rbm_raw", "rbm_reweighted"} & requested_modes:
        group_id, group_sizes = rbm_group_ids(num_nodes, rbm_batch_size, generator)
        rbm_mask = group_id[dst] == group_id[src]
        rbm_degree = torch.bincount(dst[rbm_mask], minlength=num_nodes)
        rbm_coverage = float((rbm_degree > 0).float().mean().item())
        rbm_pair_evaluations = int(num_nodes * rbm_batch_size)
        if "rbm_raw" in requested_modes:
            score_modes["rbm_raw"] = (
                full_nodes,
                rbm_mask,
                weight[rbm_mask],
                rbm_coverage,
                rbm_pair_evaluations,
            )
        if rbm_mask.any():
            sizes = group_sizes[group_id[dst[rbm_mask]]].float()
            rbm_weight = weight[rbm_mask] * (num_nodes - 1) / torch.clamp(sizes - 1, min=1.0)
        else:
            rbm_weight = weight[rbm_mask]
        if "rbm_reweighted" in requested_modes:
            score_modes["rbm_reweighted"] = (
                full_nodes,
                rbm_mask,
                rbm_weight,
                rbm_coverage,
                rbm_pair_evaluations,
            )

    return score_modes


def self_design(states: torch.Tensor, selected_nodes: torch.Tensor) -> torch.Tensor:
    selected = states[:, selected_nodes, :]
    x = selected[:, :, 0]
    y = selected[:, :, 1]
    z = selected[:, :, 2]
    return torch.stack(
        (
            torch.ones_like(x),
            x,
            x.pow(2),
            x.pow(3),
            y,
            z,
        ),
        dim=-1,
    )


def aggregate_joint_candidate_feature(
    states: torch.Tensor,
    dst: torch.Tensor,
    src: torch.Tensor,
    weight: torch.Tensor,
    selected_nodes: torch.Tensor,
    candidate: str,
) -> torch.Tensor:
    x = states[:, :, 0]
    xi = x[:, dst]
    xj = x[:, src]
    if candidate == "true_sigmoid_product":
        edge_feature = (2.0 - xi) * torch.sigmoid(xj)
    elif candidate == "product_linear":
        edge_feature = (2.0 - xi) * xj
    elif candidate == "additive_sigmoid":
        edge_feature = (2.0 - xi) + torch.sigmoid(xj)
    elif candidate == "product_square":
        edge_feature = (2.0 - xi) * xj.pow(2)
    else:
        raise ValueError(f"Unknown FEX candidate: {candidate}")

    weighted_feature = edge_feature * weight.view(1, -1)
    full_feature = torch.zeros(x.size(0), x.size(1), dtype=x.dtype)
    full_feature.scatter_add_(1, dst.view(1, -1).expand(x.size(0), -1), weighted_feature)
    return full_feature[:, selected_nodes]


def fit_joint_candidate(
    states: torch.Tensor,
    dx_target: torch.Tensor,
    dst: torch.Tensor,
    src: torch.Tensor,
    weight: torch.Tensor,
    selected_nodes: torch.Tensor,
    candidate: str,
    ridge: float,
) -> tuple[torch.Tensor, float, float, float]:
    start = time.perf_counter()
    interaction_feature = aggregate_joint_candidate_feature(
        states,
        dst,
        src,
        weight,
        selected_nodes,
        candidate,
    )
    design = torch.cat(
        (
            self_design(states, selected_nodes),
            interaction_feature.unsqueeze(-1),
        ),
        dim=-1,
    )
    flat_design = design.reshape(-1, design.size(-1)).to(torch.float64)
    flat_target = dx_target[:, selected_nodes].reshape(-1).to(torch.float64)
    gram = flat_design.T @ flat_design
    if ridge > 0.0:
        gram = gram + ridge * torch.eye(gram.size(0), dtype=gram.dtype)
    rhs = flat_design.T @ flat_target
    coeff = torch.linalg.solve(gram, rhs)
    pred = flat_design @ coeff
    dx_mse = float(torch.mean((pred - flat_target) ** 2).item())
    self_l2 = float(torch.linalg.norm(coeff[:-1] - TRUE_SELF_COEFF).item())
    elapsed = time.perf_counter() - start
    return coeff, dx_mse, self_l2, elapsed


def rbm_co_select_candidate(
    states: torch.Tensor,
    dx_target: torch.Tensor,
    dst: torch.Tensor,
    src: torch.Tensor,
    weight: torch.Tensor,
    num_nodes: int,
    *,
    batch_size: int,
    partitions: int,
    reweight: bool,
    ridge: float,
    generator: torch.Generator,
) -> tuple[str, float, int, float]:
    full_nodes = torch.arange(num_nodes, dtype=torch.long)
    candidate_scores: list[tuple[float, str]] = []
    retained_edges: list[int] = []
    coverages: list[float] = []

    for candidate_name in FEX_CANDIDATES:
        losses = []
        for _ in range(partitions):
            group_id, group_sizes = rbm_group_ids(num_nodes, batch_size, generator)
            rbm_mask = group_id[dst] == group_id[src]
            rbm_dst = dst[rbm_mask]
            rbm_src = src[rbm_mask]
            rbm_weight = weight[rbm_mask]
            rbm_degree = torch.bincount(rbm_dst, minlength=num_nodes)
            retained_edges.append(int(rbm_dst.numel()))
            coverages.append(float((rbm_degree > 0).float().mean().item()))
            if reweight and rbm_dst.numel() > 0:
                sizes = group_sizes[group_id[rbm_dst]].float()
                rbm_weight = rbm_weight * (num_nodes - 1) / torch.clamp(sizes - 1, min=1.0)

            _, dx_mse, _, _ = fit_joint_candidate(
                states,
                dx_target,
                rbm_dst,
                rbm_src,
                rbm_weight,
                full_nodes,
                candidate_name,
                ridge,
            )
            losses.append(dx_mse)
        candidate_scores.append((float(np.mean(losses)), candidate_name))

    candidate_scores.sort(key=lambda item: item[0])
    return (
        candidate_scores[0][1],
        candidate_scores[0][0],
        int(round(float(np.mean(retained_edges)))) if retained_edges else 0,
        float(np.mean(coverages)) if coverages else 0.0,
    )


def run_degree_diagnostic(args: argparse.Namespace) -> list[DegreeScoreRow]:
    np_rng = np.random.default_rng(args.seed + 1009)
    torch_gen = torch.Generator().manual_seed(args.seed + 1009)
    num_nodes = args.degree_nodes
    num_edges = int(round(num_nodes * args.mean_degree))
    dst, src, weight = sample_static_scale_free_edges(
        num_nodes,
        num_edges,
        np_rng,
        gamma_in=args.gamma_in,
        gamma_out=args.gamma_out,
    )

    degrees = torch.bincount(dst, minlength=num_nodes)
    low_candidates = torch.nonzero(degrees <= args.low_degree_max, as_tuple=True)[0]
    high_threshold = torch.quantile(degrees.float(), 0.9)
    high_candidates = torch.nonzero(degrees.float() >= high_threshold, as_tuple=True)[0]

    low_edges = int(degrees[low_candidates].sum().item())
    high_edges = int(degrees[high_candidates].sum().item())
    edge_budget = min(args.degree_edge_budget, low_edges, high_edges)

    low_nodes = choose_nodes_for_budget(low_candidates, degrees, edge_budget, torch_gen)
    high_nodes = choose_nodes_for_budget(high_candidates, degrees, edge_budget, torch_gen)

    x = sample_hr_x(args.degree_state_samples, num_nodes, torch_gen)
    true_full = aggregate_hr_interaction(x, dst, src, weight)

    rows: list[DegreeScoreRow] = []
    for subset_name, selected_nodes in (("degree_le_3", low_nodes), ("top_decile_hubs", high_nodes)):
        if selected_nodes.numel() == 0:
            continue
        target = true_full[:, selected_nodes]
        subset_edges = int(degrees[selected_nodes].sum().item())
        for candidate in args.degree_candidates:
            base = candidate_base(x, dst, src, selected_nodes, candidate)
            scale = fit_scale(base, target)
            pred = base * (0.0 if math.isnan(scale) else scale)
            rows.append(
                DegreeScoreRow(
                    subset=subset_name,
                    candidate=candidate,
                    nodes=int(selected_nodes.numel()),
                    edges=subset_edges,
                    fitted_scale=scale,
                    mse=float(torch.mean((pred - target) ** 2).item()),
                    smape=smape(pred, target),
                )
            )

    return rows


def run_joint_candidate_check(args: argparse.Namespace) -> list[JointCandidateRow]:
    np_rng = np.random.default_rng(args.seed + 3037)
    torch_gen = torch.Generator().manual_seed(args.seed + 3037)
    rows: list[JointCandidateRow] = []

    for num_nodes in args.joint_nodes:
        num_edges = int(round(num_nodes * args.mean_degree))
        dst, src, weight = sample_static_scale_free_edges(
            num_nodes,
            num_edges,
            np_rng,
            gamma_in=args.gamma_in,
            gamma_out=args.gamma_out,
        )
        full_nodes = torch.arange(num_nodes, dtype=torch.long)
        degrees = torch.bincount(dst, minlength=num_nodes)
        requested_modes = set(args.joint_modes)
        score_modes = build_scoring_modes(
            num_nodes,
            dst,
            src,
            weight,
            degrees,
            requested_modes,
            args.joint_edge_budget,
            args.joint_low_degree_max,
            args.rbm_batch_size,
            torch_gen,
        )

        for snr_db in args.joint_snr:
            states, derivatives = make_sparse_hr_timeseries(
                args.joint_timesteps,
                num_nodes,
                dst,
                src,
                weight,
                snr_db=snr_db,
                dt=args.joint_dt,
                generator=torch_gen,
            )
            dx_target = derivatives[:, :, 0]
            true_full_base = aggregate_joint_candidate_feature(
                states,
                dst,
                src,
                weight,
                full_nodes,
                "true_sigmoid_product",
            )
            true_full_interaction = COUPLING_STRENGTH * true_full_base

            for mode_name, (
                selected_nodes,
                edge_mask,
                score_weight,
                coverage,
                pair_evaluations,
            ) in score_modes.items():
                mode_rows: list[JointCandidateRow] = []
                score_dst = dst[edge_mask]
                score_src = src[edge_mask]
                score_true_interaction = true_full_interaction[:, selected_nodes]

                for candidate_name in FEX_CANDIDATES:
                    coeff, dx_mse, self_l2, elapsed = fit_joint_candidate(
                        states,
                        dx_target,
                        score_dst,
                        score_src,
                        score_weight,
                        selected_nodes,
                        candidate_name,
                        args.joint_ridge,
                    )

                    with torch.no_grad():
                        score_candidate_base = aggregate_joint_candidate_feature(
                            states,
                            score_dst,
                            score_src,
                            score_weight,
                            selected_nodes,
                            candidate_name,
                        )
                        full_candidate_base = aggregate_joint_candidate_feature(
                            states,
                            dst,
                            src,
                            weight,
                            full_nodes,
                            candidate_name,
                        )
                        fitted_coupling = float(coeff[-1].item())
                        score_interaction = fitted_coupling * score_candidate_base
                        full_interaction = fitted_coupling * full_candidate_base

                    mode_rows.append(
                        JointCandidateRow(
                            nodes=num_nodes,
                            snr_db=format_snr(snr_db),
                            mode=mode_name,
                            candidate=candidate_name,
                            score_rank=0,
                            score_nodes=int(selected_nodes.numel()),
                            score_edges=int(score_dst.numel()),
                            pair_evaluations=pair_evaluations,
                            full_edges=int(dst.numel()),
                            coverage_fraction=coverage,
                            dx_mse=dx_mse,
                            score_interaction_smape=smape(
                                score_interaction,
                                score_true_interaction,
                            ),
                            full_interaction_smape=smape(
                                full_interaction,
                                true_full_interaction,
                            ),
                            fitted_coupling=fitted_coupling,
                            coupling_ratio=fitted_coupling / COUPLING_STRENGTH,
                            self_coeff_l2=self_l2,
                            elapsed_seconds=elapsed,
                        )
                    )

                mode_rows.sort(key=lambda row: row.dx_mse)
                for rank, row in enumerate(mode_rows, start=1):
                    row.score_rank = rank
                rows.extend(mode_rows)

                if mode_name in args.joint_post_full_finetune_modes and mode_rows:
                    selected_candidate = mode_rows[0].candidate
                    co_pair_evaluations = pair_evaluations
                    co_score_edges = int(score_dst.numel())
                    co_coverage = coverage
                    if mode_name in {"rbm_raw", "rbm_reweighted"} and args.joint_rbm_co_partitions > 1:
                        selected_candidate, _, co_score_edges, co_coverage = rbm_co_select_candidate(
                            states,
                            dx_target,
                            dst,
                            src,
                            weight,
                            num_nodes,
                            batch_size=args.rbm_batch_size,
                            partitions=args.joint_rbm_co_partitions,
                            reweight=mode_name == "rbm_reweighted",
                            ridge=args.joint_ridge,
                            generator=torch_gen,
                        )
                        co_pair_evaluations = int(
                            args.joint_rbm_co_partitions * num_nodes * args.rbm_batch_size
                        )
                    coeff, dx_mse, self_l2, elapsed = fit_joint_candidate(
                        states,
                        dx_target,
                        dst,
                        src,
                        weight,
                        full_nodes,
                        selected_candidate,
                        args.joint_ridge,
                    )
                    with torch.no_grad():
                        full_candidate_base = aggregate_joint_candidate_feature(
                            states,
                            dst,
                            src,
                            weight,
                            full_nodes,
                            selected_candidate,
                        )
                        fitted_coupling = float(coeff[-1].item())
                        full_interaction = fitted_coupling * full_candidate_base

                    rows.append(
                        JointCandidateRow(
                            nodes=num_nodes,
                            snr_db=format_snr(snr_db),
                            mode=f"{mode_name}_full_finetune",
                            candidate=selected_candidate,
                            score_rank=1,
                            score_nodes=int(full_nodes.numel()),
                            score_edges=co_score_edges,
                            pair_evaluations=co_pair_evaluations + int(dst.numel()),
                            full_edges=int(dst.numel()),
                            coverage_fraction=co_coverage,
                            dx_mse=dx_mse,
                            score_interaction_smape=smape(
                                full_interaction,
                                true_full_interaction,
                            ),
                            full_interaction_smape=smape(
                                full_interaction,
                                true_full_interaction,
                            ),
                            fitted_coupling=fitted_coupling,
                            coupling_ratio=fitted_coupling / COUPLING_STRENGTH,
                            self_coeff_l2=self_l2,
                            elapsed_seconds=elapsed,
                        )
                    )

    return rows


def write_rows(path: Path, rows: list[object]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def print_scaling_summary(rows: list[ScalingRow]) -> None:
    print("\nRBM HR interaction scaling")
    print("N        cov>=1    ret_deg   sMAPE raw   sMAPE rew   scale raw   scale avg   expected")
    for num_nodes in sorted({row.nodes for row in rows}):
        subset = [row for row in rows if row.nodes == num_nodes]
        mean = lambda name: float(np.mean([getattr(row, name) for row in subset]))
        print(
            f"{num_nodes:<8d} "
            f"{mean('coverage_fraction'):<9.4f} "
            f"{mean('mean_retained_degree'):<9.4f} "
            f"{mean('smape_unweighted'):<11.2f} "
            f"{mean('smape_reweighted'):<11.2f} "
            f"{mean('fit_scale_unweighted'):<11.2f} "
            f"{mean('fit_scale_partition_average'):<11.2f} "
            f"{mean('expected_inverse_keep_scale'):<9.2f}"
        )


def print_degree_summary(rows: list[DegreeScoreRow]) -> None:
    if not rows:
        return
    print("\nDegree-stratified candidate scores")
    print("subset          candidate        edges   scale      sMAPE      MSE")
    for row in sorted(rows, key=lambda item: (item.subset, item.smape)):
        print(
            f"{row.subset:<15} {row.candidate:<16} "
            f"{row.edges:<7d} {row.fitted_scale:<10.3f} "
            f"{row.smape:<10.2f} {row.mse:.6e}"
        )


def print_fex_stratified_summary(rows: list[FEXStratifiedRow]) -> None:
    if not rows:
        return
    print("\nFEX stratified HR candidate check")
    print(
        "N      mode             rank candidate              score MSE  "
        "score sMAPE full sMAPE work-x  edges  pairs    cov    sec"
    )
    for row in sorted(rows, key=lambda item: (item.nodes, item.mode, item.score_rank)):
        work_reduction = row.full_edges / max(row.pair_evaluations, 1)
        print(
            f"{row.nodes:<6d} "
            f"{row.mode:<16} "
            f"{row.score_rank:<4d} "
            f"{row.candidate:<22} "
            f"{row.score_mse:<10.2e} "
            f"{row.score_smape:<11.2f} "
            f"{row.full_smape:<10.2f} "
            f"{work_reduction:<7.3g} "
            f"{row.score_edges:<6d} "
            f"{row.pair_evaluations:<8d} "
            f"{row.coverage_fraction:<6.3f} "
            f"{row.elapsed_seconds:.2f}"
        )


def print_joint_candidate_summary(rows: list[JointCandidateRow]) -> None:
    if not rows:
        return
    print("\nJoint HR candidate check with five-point derivatives")
    print(
        "N      SNR   mode             rank candidate              dx MSE     "
        "score int full int  c/c*    self L2 work-x  pairs"
    )
    ordered = sorted(rows, key=lambda item: (item.snr_db, item.nodes, item.mode, item.score_rank))
    for row in ordered:
        work_reduction = row.full_edges / max(row.pair_evaluations, 1)
        print(
            f"{row.nodes:<6d} "
            f"{row.snr_db:<5} "
            f"{row.mode:<16} "
            f"{row.score_rank:<4d} "
            f"{row.candidate:<22} "
            f"{row.dx_mse:<10.2e} "
            f"{row.score_interaction_smape:<9.2f} "
            f"{row.full_interaction_smape:<9.2f} "
            f"{row.coupling_ratio:<7.2f} "
            f"{row.self_coeff_l2:<7.2f} "
            f"{work_reduction:<7.3g} "
            f"{row.pair_evaluations:<8d}"
        )


def plot_fex_stratified_comparison(rows: list[FEXStratifiedRow], path: Path) -> None:
    if not rows:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping FEX comparison plot.")
        return

    labels = {
        "full": "Full",
        "stratified_low": "Stratified low",
        "top_decile_hubs": "Hubs",
        "rbm_raw": "RBM",
        "rbm_reweighted": "RBM reweighted",
    }
    colors = {
        "full": "black",
        "stratified_low": "green",
        "top_decile_hubs": "tab:purple",
        "rbm_raw": "orange",
        "rbm_reweighted": "blue",
    }
    markers = {
        "full": "o",
        "stratified_low": "D",
        "top_decile_hubs": "P",
        "rbm_raw": "s",
        "rbm_reweighted": "^",
    }

    best_rows = [row for row in rows if row.score_rank == 1]
    modes = [mode for mode in labels if any(row.mode == mode for row in best_rows)]
    nodes = sorted({row.nodes for row in best_rows})

    fig, (ax_recovery, ax_work) = plt.subplots(1, 2, figsize=(9.8, 4.4))

    for mode in modes:
        mode_rows = {row.nodes: row for row in best_rows if row.mode == mode}
        xs = [node for node in nodes if node in mode_rows]
        if not xs:
            continue
        ys_recovery = [mode_rows[node].full_smape for node in xs]
        ys_speedup = [
            mode_rows[node].full_edges / max(mode_rows[node].pair_evaluations, 1)
            for node in xs
        ]

        ax_recovery.plot(
            xs,
            ys_recovery,
            marker=markers[mode],
            color=colors[mode],
            linewidth=2,
            markersize=5,
            label=labels[mode],
        )
        ax_work.plot(
            xs,
            ys_speedup,
            marker=markers[mode],
            color=colors[mode],
            linewidth=2,
            markersize=5,
            label=labels[mode],
        )

    ax_recovery.set_xscale("log")
    ax_recovery.set_xlabel(r"Network Size $n$")
    ax_recovery.set_ylabel("Full-Graph Interaction sMAPE (%)")
    ax_recovery.grid(alpha=0.22, linestyle="--")
    ax_recovery.legend(fontsize=8, frameon=False)
    ax_recovery.text(-0.18, 0.98, "(a)", transform=ax_recovery.transAxes, va="top", fontsize=13)

    ax_work.set_xscale("log")
    ax_work.set_yscale("log")
    ax_work.set_xlabel(r"Network Size $n$")
    ax_work.set_ylabel(r"Sparse-Full / Pair Evaluations ($\times$)")
    ax_work.grid(alpha=0.22, linestyle="--")
    ax_work.axhline(1.0, color="black", linewidth=1, linestyle=":", alpha=0.7)
    ax_work.text(-0.18, 0.98, "(b)", transform=ax_work.transAxes, va="top", fontsize=13)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_joint_candidate_single_snr(
    rows: list[JointCandidateRow],
    path: Path,
    snr_label: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping joint candidate plot.")
        return

    labels = {
        "full": "Full",
        "stratified_low": "Stratified low",
        "top_decile_hubs": "Hubs",
        "rbm_raw": "RBM-CO",
        "rbm_reweighted": "RBM-CO reweighted",
        "rbm_raw_full_finetune": "RBM-CO + full fit",
        "rbm_reweighted_full_finetune": "RBM-IPW-CO + full fit",
    }
    colors = {
        "full": "black",
        "stratified_low": "green",
        "top_decile_hubs": "tab:purple",
        "rbm_raw": "orange",
        "rbm_reweighted": "blue",
        "rbm_raw_full_finetune": "darkorange",
        "rbm_reweighted_full_finetune": "royalblue",
    }
    markers = {
        "full": "o",
        "stratified_low": "D",
        "top_decile_hubs": "P",
        "rbm_raw": "s",
        "rbm_reweighted": "^",
        "rbm_raw_full_finetune": "X",
        "rbm_reweighted_full_finetune": "v",
    }

    best_rows = [row for row in rows if row.score_rank == 1 and row.snr_db == snr_label]
    if not best_rows:
        return
    modes = [mode for mode in labels if any(row.mode == mode for row in best_rows)]
    nodes = sorted({row.nodes for row in best_rows})

    fig, (ax_recovery, ax_work) = plt.subplots(1, 2, figsize=(9.8, 4.4))

    for mode in modes:
        mode_rows = {row.nodes: row for row in best_rows if row.mode == mode}
        xs = [node for node in nodes if node in mode_rows]
        if not xs:
            continue
        ys_recovery = [mode_rows[node].full_interaction_smape for node in xs]
        ys_work = [
            mode_rows[node].full_edges / max(mode_rows[node].pair_evaluations, 1)
            for node in xs
        ]

        ax_recovery.plot(
            xs,
            ys_recovery,
            marker=markers[mode],
            color=colors[mode],
            linewidth=2,
            markersize=5,
            label=labels[mode],
        )
        ax_work.plot(
            xs,
            ys_work,
            marker=markers[mode],
            color=colors[mode],
            linewidth=2,
            markersize=5,
            label=labels[mode],
        )

    ax_recovery.set_xscale("log")
    ax_recovery.set_xlabel(r"Network Size $n$")
    ax_recovery.set_ylabel("Full-Graph Interaction sMAPE (%)")
    ax_recovery.set_title(f"SNR {snr_label}")
    ax_recovery.grid(alpha=0.22, linestyle="--")
    ax_recovery.legend(fontsize=8, frameon=False)
    ax_recovery.text(-0.18, 0.98, "(a)", transform=ax_recovery.transAxes, va="top", fontsize=13)

    ax_work.set_xscale("log")
    ax_work.set_yscale("log")
    ax_work.set_xlabel(r"Network Size $n$")
    ax_work.set_ylabel(r"Sparse-Full / Pair Evaluations ($\times$)")
    ax_work.grid(alpha=0.22, linestyle="--")
    ax_work.axhline(1.0, color="black", linewidth=1, linestyle=":", alpha=0.7)
    ax_work.text(-0.18, 0.98, "(b)", transform=ax_work.transAxes, va="top", fontsize=13)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_joint_candidate_comparison(rows: list[JointCandidateRow], path: Path) -> None:
    snr_labels = sorted(
        {row.snr_db for row in rows},
        key=lambda item: float("inf") if item == "inf" else float(item),
    )
    if not snr_labels:
        return
    if len(snr_labels) == 1:
        _plot_joint_candidate_single_snr(rows, path, snr_labels[0])
        return

    stem = path.with_suffix("")
    for snr_label in snr_labels:
        _plot_joint_candidate_single_snr(
            rows,
            stem.parent / f"{stem.name}_snr_{snr_label}.png",
            snr_label,
        )


def plot_scaling(rows: list[ScalingRow], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plot.")
        return

    nodes = sorted({row.nodes for row in rows})

    def mean_and_std(name: str) -> tuple[np.ndarray, np.ndarray]:
        means = []
        stds = []
        for num_nodes in nodes:
            values = np.array([getattr(row, name) for row in rows if row.nodes == num_nodes], dtype=float)
            means.append(values.mean())
            stds.append(values.std(ddof=1) if values.size > 1 else 0.0)
        return np.array(means), np.array(stds)

    smape_raw, smape_raw_std = mean_and_std("smape_unweighted")
    smape_rew, smape_rew_std = mean_and_std("smape_reweighted")
    coverage, coverage_std = mean_and_std("coverage_fraction")

    fig, (ax_loss, ax_cov) = plt.subplots(1, 2, figsize=(9.5, 3.8))
    ax_loss.errorbar(nodes, smape_raw, yerr=smape_raw_std, marker="o", label="Unweighted RBM")
    ax_loss.errorbar(nodes, smape_rew, yerr=smape_rew_std, marker="s", label="Reweighted RBM")
    ax_loss.set_xscale("log")
    ax_loss.set_xlabel("Network size N")
    ax_loss.set_ylabel("Interaction sMAPE (%)")
    ax_loss.grid(alpha=0.25, linestyle="--")
    ax_loss.legend(frameon=False)

    ax_cov.errorbar(nodes, coverage, yerr=coverage_std, marker="o", color="tab:red")
    ax_cov.set_xscale("log")
    ax_cov.set_xlabel("Network size N")
    ax_cov.set_ylabel("Fraction with retained neighbor")
    ax_cov.grid(alpha=0.25, linestyle="--")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=parse_node_list, default=parse_node_list("512,2000,5000,10000"))
    parser.add_argument("--mean_degree", type=float, default=6.0)
    parser.add_argument("--rbm_batch_size", type=int, default=32)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--state_samples", type=int, default=8)
    parser.add_argument("--scale_partitions", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma_in", type=float, default=5.0)
    parser.add_argument("--gamma_out", type=float, default=5.0)
    parser.add_argument("--skip_rbm_scaling", action="store_true")
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("NumericalExperiments/HR/logs_xingjian_test"),
    )
    parser.add_argument("--skip_degree_diagnostic", action="store_true")
    parser.add_argument("--degree_nodes", type=int, default=512)
    parser.add_argument("--degree_state_samples", type=int, default=128)
    parser.add_argument("--degree_edge_budget", type=int, default=512)
    parser.add_argument("--low_degree_max", type=int, default=3)
    parser.add_argument(
        "--degree_candidates",
        nargs="+",
        default=["sigmoid_true", "constant_mean", "affine_sigmoid", "quadratic"],
    )
    parser.add_argument("--skip_fex_stratified", action="store_true")
    parser.add_argument("--fex_nodes", type=parse_node_list, default=parse_node_list("512"))
    parser.add_argument("--fex_state_samples", type=int, default=64)
    parser.add_argument("--fex_edge_budget", type=int, default=512)
    parser.add_argument("--fex_low_degree_max", type=int, default=3)
    parser.add_argument("--fex_epochs", type=int, default=120)
    parser.add_argument("--fex_lr", type=float, default=0.03)
    parser.add_argument("--fex_restarts", type=int, default=1)
    parser.add_argument(
        "--fex_modes",
        nargs="+",
        default=["full", "stratified_low", "top_decile_hubs", "rbm_raw", "rbm_reweighted"],
        choices=["full", "stratified_low", "top_decile_hubs", "rbm_raw", "rbm_reweighted"],
    )
    parser.add_argument("--fex_stop_mse", type=float, default=1e-12)
    parser.add_argument(
        "--fex_hr_warm_start",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--skip_joint_candidate", action="store_true")
    parser.add_argument("--joint_nodes", type=parse_node_list, default=parse_node_list("512,2000,5000"))
    parser.add_argument("--joint_timesteps", type=int, default=160)
    parser.add_argument("--joint_dt", type=float, default=0.01)
    parser.add_argument("--joint_snr", type=parse_snr_list, default=parse_snr_list("inf,45,40"))
    parser.add_argument("--joint_edge_budget", type=int, default=512)
    parser.add_argument("--joint_low_degree_max", type=int, default=3)
    parser.add_argument("--joint_ridge", type=float, default=1e-8)
    parser.add_argument(
        "--joint_modes",
        nargs="+",
        default=["full", "stratified_low", "rbm_raw", "rbm_reweighted"],
        choices=["full", "stratified_low", "top_decile_hubs", "rbm_raw", "rbm_reweighted"],
    )
    parser.add_argument(
        "--joint_post_full_finetune_modes",
        nargs="*",
        default=["rbm_raw", "rbm_reweighted"],
        choices=["rbm_raw", "rbm_reweighted"],
    )
    parser.add_argument("--joint_rbm_co_partitions", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_rbm_scaling:
        scaling_rows = run_rbm_scaling(args)
        write_rows(args.out_dir / "rbm_hr_scaling.csv", scaling_rows)
        plot_scaling(scaling_rows, args.out_dir / "rbm_hr_scaling.png")
        print_scaling_summary(scaling_rows)

    if not args.skip_degree_diagnostic:
        degree_rows = run_degree_diagnostic(args)
        write_rows(args.out_dir / "degree_candidate_scores.csv", degree_rows)
        print_degree_summary(degree_rows)

    if not args.skip_fex_stratified:
        fex_rows = run_fex_stratified_check(args)
        write_rows(args.out_dir / "fex_stratified_scores.csv", fex_rows)
        plot_fex_stratified_comparison(fex_rows, args.out_dir / "fex_stratified_comparison.png")
        print_fex_stratified_summary(fex_rows)

    if not args.skip_joint_candidate:
        joint_rows = run_joint_candidate_check(args)
        write_rows(args.out_dir / "joint_candidate_scores.csv", joint_rows)
        plot_joint_candidate_comparison(joint_rows, args.out_dir / "joint_candidate_comparison.png")
        print_joint_candidate_summary(joint_rows)

    print(f"\nSaved results to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
