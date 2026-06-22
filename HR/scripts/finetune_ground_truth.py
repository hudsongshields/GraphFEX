"""Fine-tune the paper-style FEX leaves for the known HR operator sequences."""

import argparse
import math
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from FEX.models.learnable_tree import FEX
from FEX.training.loss_funcs import total_loss
from FEX.training.train_configs import FEXConfig, runtimeconfig
from FEX.training.train_fex import train_network_fex
from FEX.utils.tree_configs import get_tree_config


SELF_SEQUENCE = [0, 0, 0, 0, 1, 2, 0]
INTERACTION_SEQUENCE = [1, 0, 3]
VARIABLES = ["x_i", "y_i", "z_i", "x_j", "y_j", "z_j"]


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_adjacency(num_nodes: int, probability: float, device) -> torch.Tensor:
    adjacency = (torch.rand(num_nodes, num_nodes) < probability).float()
    adjacency.fill_diagonal_(0.0)
    for node in range(num_nodes):
        if adjacency[node].sum() == 0:
            adjacency[node, (node + 1) % num_nodes] = 1.0
    return adjacency.to(device)


def make_data(num_samples: int, adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    num_nodes = adjacency.size(0)
    states = torch.empty(num_samples, num_nodes, 3, device=adjacency.device)
    states[..., 0].uniform_(-2.0, 2.0)
    states[..., 1].uniform_(-8.0, 4.0)
    states[..., 2].uniform_(0.0, 5.0)

    x_i = states[..., 0]
    y_i = states[..., 1]
    z_i = states[..., 2]
    self_dynamics = y_i - x_i.pow(3) + 3.0 * x_i.pow(2) - z_i + 3.24
    pairwise = 0.15 * (2.0 - x_i).unsqueeze(2) * torch.sigmoid(x_i).unsqueeze(1)
    dx = self_dynamics + (pairwise * adjacency.unsqueeze(0)).sum(dim=2)

    derivatives = torch.zeros_like(states)
    derivatives[..., 0] = dx
    return states.cpu(), derivatives.cpu()


def initialize_small(tree: FEX, std: float = 0.01) -> None:
    with torch.no_grad():
        for leaf in tree.leaf_mlps:
            leaf.logits.normal_(mean=0.0, std=std)
            leaf.bias.zero_()


def self_coefficients(tree: FEX) -> dict[str, float]:
    identity_left, square_leaf, cube_leaf, identity_right = tree.leaf_mlps
    linear = identity_left.logits.detach() + identity_right.logits.detach()
    square = square_leaf.logits.detach()
    cube = cube_leaf.logits.detach()
    return {
        "x": float(linear[0]),
        "y": float(linear[1]),
        "z": float(linear[2]),
        "x^2": float(square[0]),
        "y^2": float(square[1]),
        "z^2": float(square[2]),
        "x^3": float(cube[0]),
        "y^3": float(cube[1]),
        "z^3": float(cube[2]),
        "constant": float(sum(leaf.bias.detach().item() for leaf in tree.leaf_mlps)),
    }


def self_targets() -> dict[str, float]:
    return {
        "x": 0.0,
        "y": 1.0,
        "z": -1.0,
        "x^2": 3.0,
        "y^2": 0.0,
        "z^2": 0.0,
        "x^3": -1.0,
        "y^3": 0.0,
        "z^3": 0.0,
        "constant": 3.24,
    }


def interaction_coefficients(tree: FEX) -> dict[str, float]:
    identity_leaf, sigmoid_leaf = tree.leaf_mlps
    left_weights = identity_leaf.logits.detach()
    right_weights = sigmoid_leaf.logits.detach()
    left_bias = float(identity_leaf.bias.detach().item())
    right_bias = float(sigmoid_leaf.bias.detach().item())

    coefficients = {"constant": left_bias * right_bias}
    for index, variable in enumerate(VARIABLES):
        coefficients[variable] = float(left_weights[index]) * right_bias
        coefficients[f"sigmoid({variable})"] = left_bias * float(right_weights[index])
    for left_index, left_variable in enumerate(VARIABLES):
        for right_index, right_variable in enumerate(VARIABLES):
            key = f"{left_variable}*sigmoid({right_variable})"
            coefficients[key] = float(left_weights[left_index] * right_weights[right_index])
    return coefficients


def interaction_targets() -> dict[str, float]:
    targets = {"constant": 0.0}
    for variable in VARIABLES:
        targets[variable] = 0.0
        targets[f"sigmoid({variable})"] = 0.0
    for left_variable in VARIABLES:
        for right_variable in VARIABLES:
            targets[f"{left_variable}*sigmoid({right_variable})"] = 0.0
    targets["sigmoid(x_j)"] = 0.30
    targets["x_i*sigmoid(x_j)"] = -0.15
    return targets


def maximum_error(actual: dict[str, float], target: dict[str, float]) -> float:
    return max(abs(actual[key] - target[key]) for key in target)


def cleaned_expressions(self_values, interaction_values, threshold: float):
    import sympy as sp

    symbols = dict(zip(VARIABLES, sp.symbols(" ".join(VARIABLES))))
    sigmoid = sp.Function("sigmoid")
    x_i, y_i, z_i = symbols["x_i"], symbols["y_i"], symbols["z_i"]
    self_basis = {
        "x": x_i,
        "y": y_i,
        "z": z_i,
        "x^2": x_i**2,
        "y^2": y_i**2,
        "z^2": z_i**2,
        "x^3": x_i**3,
        "y^3": y_i**3,
        "z^3": z_i**3,
        "constant": sp.Integer(1),
    }
    self_expression = sum(
        sp.Float(round(coefficient, 4)) * self_basis[term]
        for term, coefficient in self_values.items()
        if abs(coefficient) >= threshold
    )

    interaction_expression = sp.Integer(0)
    for term, coefficient in interaction_values.items():
        if abs(coefficient) < threshold:
            continue
        rounded = sp.Float(round(coefficient, 4))
        if term == "constant":
            basis = sp.Integer(1)
        elif "*sigmoid(" in term:
            left_name, right_part = term.split("*sigmoid(", 1)
            right_name = right_part[:-1]
            basis = symbols[left_name] * sigmoid(symbols[right_name])
        elif term.startswith("sigmoid("):
            variable_name = term[len("sigmoid("):-1]
            basis = sigmoid(symbols[variable_name])
        else:
            basis = symbols[term]
        interaction_expression += rounded * basis

    sigmoid_terms = [sigmoid(symbols[name]) for name in VARIABLES]
    readable_interaction = sp.collect(sp.expand(interaction_expression), sigmoid_terms)
    return sp.simplify(self_expression), readable_interaction


def print_nonzero_comparison(title: str, actual: dict[str, float], target: dict[str, float]) -> None:
    print(f"\n{title}")
    print(f"{'term':<28} {'recovered':>12} {'target':>12} {'abs error':>12}")
    for term, target_value in target.items():
        recovered = actual[term]
        if target_value != 0.0 or abs(recovered) >= 1e-3:
            print(
                f"{term:<28} {recovered:>12.6f} {target_value:>12.6f} "
                f"{abs(recovered - target_value):>12.6f}"
            )


def evaluate_mse(forcing_tree, interaction_tree, dataloader, adjacency, device) -> float:
    forcing_tree.eval()
    interaction_tree.eval()
    nodes, edges = adjacency.nonzero(as_tuple=True)
    non_self = nodes != edges
    nodes = nodes[non_self].to(device)
    edges = edges[non_self].to(device)
    groups = torch.arange(adjacency.size(0), device=device)
    scatter_idx = (nodes.unsqueeze(1) == groups.unsqueeze(0)).int().argmax(dim=1)

    losses = []
    with torch.no_grad():
        for batch_x, batch_dy in dataloader:
            losses.append(
                total_loss(
                    batch_x.to(device),
                    batch_dy[..., 0:1].to(device),
                    forcing_tree,
                    interaction_tree,
                    nodes,
                    edges,
                    scatter_idx,
                ).item()
            )
    return sum(losses) / max(len(losses), 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--bfgs-epochs", type=int, default=100)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--nodes", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--coefficient-tolerance", type=float, default=0.02)
    parser.add_argument("--mse-tolerance", type=float, default=1e-5)
    parser.add_argument("--expression-threshold", type=float, default=0.002)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(runtimeconfig.device)
    adjacency = make_adjacency(args.nodes, probability=0.35, device=device)
    states, derivatives = make_data(args.samples, adjacency)
    dataloader = DataLoader(
        TensorDataset(states, derivatives),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )

    self_structure = get_tree_config("depth_3_leaves_4_config")
    interaction_structure = get_tree_config("depth_2_tree_config")
    forcing_tree = FEX(
        sample_indices=SELF_SEQUENCE,
        leaf_dim=3,
        num_leaves=self_structure.num_leaves,
        tree_structure=self_structure,
    ).to(device)
    interaction_tree = FEX(
        sample_indices=INTERACTION_SEQUENCE,
        leaf_dim=6,
        num_leaves=interaction_structure.num_leaves,
        tree_structure=interaction_structure,
    ).to(device)
    initialize_small(forcing_tree)
    initialize_small(interaction_tree)

    config = FEXConfig(
        num_epochs=args.epochs,
        bfgs_epochs=args.bfgs_epochs,
        lr=args.lr,
        inter_lr=args.lr,
        bfgs_lr=0.5,
        leaf_dim=3,
        num_leaves=self_structure.num_leaves,
        leaf_entropy_weight=0.0,
        mag_entropy_weight=0.0,
        tau_start=1.0,
        tau_end=1.0,
        tau_schedule="non_decay",
    )

    print(f"Device: {device}")
    print(f"Self operator sequence: {SELF_SEQUENCE}")
    print(f"Interaction operator sequence: {INTERACTION_SEQUENCE}")
    print("Interaction masks: disabled")
    print(
        f"Fine-tuning: Adam epochs={args.epochs}, BFGS iterations={args.bfgs_epochs}, "
        f"samples={args.samples}"
    )
    score = train_network_fex(
        forcing_tree,
        interaction_tree,
        dataloader,
        adjacency,
        config,
        device=device,
        log_every=args.log_every,
    )

    recovered_self = self_coefficients(forcing_tree)
    recovered_interaction = interaction_coefficients(interaction_tree)
    target_self = self_targets()
    target_interaction = interaction_targets()
    mse = evaluate_mse(forcing_tree, interaction_tree, dataloader, adjacency, device)
    self_error = maximum_error(recovered_self, target_self)
    interaction_error = maximum_error(recovered_interaction, target_interaction)
    coefficient_error = max(self_error, interaction_error)

    print_nonzero_comparison("Self dynamics", recovered_self, target_self)
    print_nonzero_comparison("Interaction dynamics", recovered_interaction, target_interaction)
    print(f"\nReturned score: {score:.8e}")
    print(f"Evaluation MSE: {mse:.8e}")
    clean_self, clean_interaction = cleaned_expressions(
        recovered_self,
        recovered_interaction,
        args.expression_threshold,
    )
    print(f"Cleaned SymPy threshold: {args.expression_threshold:g}")
    print("Self SymPy:", clean_self)
    print("Interaction SymPy:", clean_interaction)
    print(f"Maximum self coefficient error: {self_error:.8e}")
    print(f"Maximum interaction coefficient error: {interaction_error:.8e}")

    passed = (
        math.isfinite(mse)
        and mse <= args.mse_tolerance
        and coefficient_error <= args.coefficient_tolerance
    )
    print("RESULT:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
