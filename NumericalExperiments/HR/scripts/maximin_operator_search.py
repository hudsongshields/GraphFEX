"""One-process maximin search for the Hindmarsh-Rose operator sequences.

Stops when one candidate has both the self operators and the interaction
operators. Add/sub may be swapped, and the four self functions may sit on any
leaves. Identity and sigmoid may sit on either interaction leaf.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from FEX.helpers.sampler import epsilon_greedy_sample
from FEX.helpers.tree_configs import get_tree_config
from FEX.models.controllers import Controller
from FEX.models.learnable_tree import FEX
from FEX.training.train_configs import ControllerConfig, FEXConfig
from FEX.training.train_fex import train_network_fex

BINARY = {0: "add", 1: "mul", 2: "sub"}
UNARY = {0: "id", 1: "square", 2: "cube", 3: "fourth", 4: "exp", 5: "sigmoid", 6: "sin"}
EXACT_SELF = (0, 0, 0, 0, 2, 1, 0)
EXACT_INTER = (1, 0, 5)


def make_static_sf_adjacency(num_nodes, num_edges, gamma_in=3.5, gamma_out=3.5):
    alpha_in = 1.0 / (gamma_in - 1.0)
    alpha_out = 1.0 / (gamma_out - 1.0)
    idx = 1.0 + np.arange(num_nodes)
    weights_in = np.random.permutation(1.0 / (idx ** alpha_in))
    weights_out = np.random.permutation(1.0 / (idx ** alpha_out))
    prob_in = weights_in / np.sum(weights_in)
    prob_out = weights_out / np.sum(weights_out)
    sampled_sources = np.random.choice(num_nodes, size=num_edges * 2, p=prob_out)
    sampled_targets = np.random.choice(num_nodes, size=num_edges * 2, p=prob_in)
    unique_edges = set()
    for source, target in zip(sampled_sources, sampled_targets):
        if source != target:
            unique_edges.add((int(target), int(source)))
        if len(unique_edges) >= num_edges:
            break
    adjacency = torch.zeros(num_nodes, num_nodes)
    if unique_edges:
        edges_idx = torch.tensor(list(unique_edges), dtype=torch.long)
        adjacency[edges_idx[:, 0], edges_idx[:, 1]] = 1.0
    return adjacency


def hindmarsh_rose_rhs(state, adjacency):
    x_state, y_state, z_state = state[:, 0], state[:, 1], state[:, 2]
    interaction = 0.15 * (2.0 - x_state) * (adjacency @ torch.sigmoid(x_state))
    dx_dt = y_state - x_state.pow(3) + 3.0 * x_state.pow(2) - z_state + 3.24 + interaction
    dy_dt = 1.0 - 5.0 * x_state.pow(2) - y_state
    dz_dt = 0.005 * (4.0 * (x_state + 1.6) - z_state)
    return torch.stack((dx_dt, dy_dt, dz_dt), dim=-1)


def rk4_step(state, adjacency, dt):
    k1 = hindmarsh_rose_rhs(state, adjacency)
    k2 = hindmarsh_rose_rhs(state + 0.5 * dt * k1, adjacency)
    k3 = hindmarsh_rose_rhs(state + 0.5 * dt * k2, adjacency)
    k4 = hindmarsh_rose_rhs(state + dt * k3, adjacency)
    return state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def five_point(timeseries, dt):
    return (timeseries[:-4] - 8 * timeseries[1:-3] + 8 * timeseries[3:-1] - timeseries[4:]) / (12 * dt)


def make_timeseries(num_samples, adjacency, snr, dt=0.01):
    states = torch.empty(num_samples, adjacency.size(0), 3)
    states[0, :, 0].uniform_(-2.0, 2.0)
    states[0, :, 1].uniform_(-8.0, 4.0)
    states[0, :, 2].uniform_(0.0, 5.0)
    with torch.no_grad():
        for step in range(1, num_samples):
            states[step] = rk4_step(states[step - 1], adjacency, dt)
    noise_power = torch.var(states, dim=0, correction=1).mean() / (10.0 ** (snr / 10.0))
    observed = states + torch.sqrt(noise_power) * torch.randn_like(states)
    return observed[2:-2], five_point(observed, dt)


def as_tuple(indices):
    return tuple(int(value) for value in indices.detach().cpu().tolist())


def self_equivalent(seq):
    return all(op in (0, 2) for op in seq[:3]) and sorted(seq[3:]) == [0, 0, 1, 2]


def inter_equivalent(seq):
    return seq[0] == 1 and sorted(seq[1:]) == [0, 5]


def describe(self_seq, inter_seq):
    self_text = f"{[BINARY[i] for i in self_seq[:3]]}/{[UNARY[i] for i in self_seq[3:]]}"
    inter_text = f"{BINARY[inter_seq[0]]}({UNARY[inter_seq[1]]}, {UNARY[inter_seq[2]]})"
    return self_text, inter_text


def tree_expression(tree):
    try:
        return tree.simplified_expression()
    except Exception as exc:
        return f"{tree.expression_summary()} [expression failed: {exc}]"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snr", type=float, default=30.0)
    parser.add_argument("--nodes", type=int, default=100)
    parser.add_argument("--edges", type=int, default=500)
    parser.add_argument("--timesteps", type=int, default=800)
    parser.add_argument("--num-groups", type=int, default=4)
    parser.add_argument("--fex-epochs", type=int, default=8)
    parser.add_argument("--cands-per-epoch", type=int, default=4)
    parser.add_argument("--limit-hours", type=float, default=9.75)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("NumericalExperiments/HR/slurm/maximin_operator_search_summary.txt"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(1)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no GPU is visible")

    adjacency = make_static_sf_adjacency(args.nodes, args.edges)
    series, derivs = make_timeseries(args.timesteps, adjacency, snr=args.snr)
    dataloader = DataLoader(TensorDataset(series, derivs), batch_size=128, shuffle=True)
    self_struct = get_tree_config("depth_3_leaves_4_config")
    inter_struct = get_tree_config("depth_2_tree_config")
    ops_per_node = self_struct.ops_per_node + inter_struct.ops_per_node

    controller_config = ControllerConfig(lr=0.003, epsilon_greedy=0.2, percentile_threshold=0.4)
    fex_config = FEXConfig(
        target_dim=0,
        lr=0.02,
        inter_lr=0.02,
        num_epochs=args.fex_epochs,
        bfgs_epochs=0,
        expression_threshold=0.1,
    )
    controller = Controller(
        ops_per_node=ops_per_node,
        input_size=controller_config.input_dim,
        hidden_size=controller_config.hidden_dim,
    ).to(args.device)
    optimizer = torch.optim.Adam(controller.parameters(), lr=controller_config.lr)
    controller_input = torch.zeros(controller_config.input_dim, device=args.device)

    print(
        f"SEARCH device={args.device} maximin groups={args.num_groups} snr={args.snr} "
        f"limit_hours={args.limit_hours} fex_epochs={args.fex_epochs}",
        flush=True,
    )
    print(f"TARGET self {list(EXACT_SELF)} inter {list(EXACT_INTER)}", flush=True)

    started = time.perf_counter()
    deadline = started + args.limit_hours * 3600.0
    best = None
    n_self = n_inter = n_both = n_exact = 0
    seen = 0
    found = None
    epoch = 0

    while time.perf_counter() < deadline and found is None:
        epoch += 1
        optimizer.zero_grad()
        pmfs = controller(controller_input)
        paired = []
        for _ in range(args.cands_per_epoch):
            if time.perf_counter() >= deadline:
                break
            op_indices = epsilon_greedy_sample(pmfs, epsilon=controller_config.epsilon_greedy).squeeze(0)
            chosen = torch.stack([pmfs[i].squeeze(0)[op_indices[i]] for i in range(op_indices.numel())])
            log_prob = torch.log(chosen).sum()
            self_idx = op_indices[: len(self_struct.ops_per_node)].cpu()
            inter_idx = op_indices[len(self_struct.ops_per_node) :].cpu()
            forcing = FEX(leaf_dim=3, sample_indices=self_idx, tree_structure=self_struct, expression_threshold=0.1)
            inter = FEX(leaf_dim=6, sample_indices=inter_idx, tree_structure=inter_struct, expression_threshold=0.1)
            loss = train_network_fex(
                forcing,
                inter,
                dataloader,
                adjacency,
                fex_config,
                num_groups=args.num_groups,
                grouping="maximin",
                device=args.device,
            )
            reward = 0.0 if not np.isfinite(loss) else 1.0 / (1.0 + float(loss))
            self_seq = as_tuple(self_idx)
            inter_seq = as_tuple(inter_idx)
            seen += 1
            self_ok = self_equivalent(self_seq)
            inter_ok = inter_equivalent(inter_seq)
            n_self += int(self_ok)
            n_inter += int(inter_ok)
            n_both += int(self_ok and inter_ok)
            n_exact += int(self_seq == EXACT_SELF and inter_seq == EXACT_INTER)
            keep_expression = self_ok or inter_ok or best is None or reward > best[0]
            self_expr = tree_expression(forcing) if keep_expression else ""
            inter_expr = tree_expression(inter) if keep_expression else ""
            row = (reward, float(loss), self_seq, inter_seq, self_expr, inter_expr)
            paired.append((log_prob, row))
            if best is None or reward > best[0]:
                best = row
            if self_ok:
                self_text, _ = describe(self_seq, inter_seq)
                print(
                    f"SELF_HIT seen={seen} seq={list(self_seq)} ops={self_text} loss={float(loss):.3f}",
                    flush=True,
                )
                print(f"SELF_EXPR {self_expr}", flush=True)
            if inter_ok:
                _, inter_text = describe(self_seq, inter_seq)
                print(
                    f"INTER_HIT seen={seen} seq={list(inter_seq)} ops={inter_text} loss={float(loss):.3f}",
                    flush=True,
                )
                print(f"INTER_EXPR {inter_expr}", flush=True)
            if self_ok and inter_ok:
                found = row
                break
        if not paired:
            break
        paired.sort(key=lambda item: item[1][0], reverse=True)
        keep = max(1, int(controller_config.percentile_threshold * len(paired)))
        top = paired[:keep]
        rewards = torch.tensor([item[1][0] for item in top], device=args.device)
        advantage = rewards - rewards.min()
        policy_loss = -(advantage * torch.stack([item[0] for item in top])).mean()
        policy_loss.backward()
        optimizer.step()
        if epoch % 25 == 0 or found is not None:
            elapsed = time.perf_counter() - started
            print(
                f"epoch {epoch} seen={seen} sec={elapsed:.1f} "
                f"self_hits={n_self} inter_hits={n_inter} both={n_both} exact={n_exact} "
                f"best_loss={best[1]:.3f}",
                flush=True,
            )

    reward, loss, self_seq, inter_seq, self_expr, inter_expr = found if found is not None else best
    self_text, inter_text = describe(self_seq, inter_seq)
    elapsed = time.perf_counter() - started
    status = "FOUND" if found is not None else "NOT_FOUND"
    lines = [
        f"{status}",
        f"seen={seen} sec={elapsed:.1f} device={args.device}",
        f"self_hits={n_self} inter_hits={n_inter} both={n_both} exact={n_exact}",
        f"SELF_SEQ {list(self_seq)} {self_text}",
        f"SELF_EXPR {self_expr}",
        f"INTER_SEQ {list(inter_seq)} {inter_text}",
        f"INTER_EXPR {inter_expr}",
        f"LOSS {loss:.6e} REWARD {reward:.4f}",
    ]
    print("\n".join(lines), flush=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
