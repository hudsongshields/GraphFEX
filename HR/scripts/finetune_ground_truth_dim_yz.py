"""Fine-tune the paper-style FEX leaves for the known HR operator sequences."""

import argparse
import math
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader, TensorDataset
from HR.data.generate_data import make_adjacency, make_data


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from FEX.models.learnable_tree import FEX
from FEX.training.loss_funcs import total_loss
from FEX.training.train_configs import FEXConfig, runtimeconfig
from FEX.training.train_fex import train_fex
from FEX.utils.tree_configs import get_tree_config


SELF_SEQUENCE = [0, 0, 1]
VARIABLES = ["x_i", "y_i", "z_i", "x_j", "y_j", "z_j"]


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



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



def evaluate_mse(forcing_tree, dataloader, device, dim) -> float:
    forcing_tree.eval()

    losses = []
    with torch.no_grad():
        for batch_x, batch_dy in dataloader:
            losses.append(
                total_loss(batch_x.to(device), batch_dy[..., dim:dim+1].to(device), forcing_tree).item()
            )
    return sum(losses) / max(len(losses), 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--bfgs-epochs", type=int, default=0)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--nodes", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--coefficient-tolerance", type=float, default=0.02)
    parser.add_argument("--mse-tolerance", type=float, default=1e-5)
    parser.add_argument("--expression-threshold", type=float, default=0.002)
    parser.add_argument("--dim", type=int, default=1)
    args = parser.parse_args()
    if args.dim == 2:
        SELF_SEQUENCE = [0, 0, 5]
    if args.dim == 1:
        SELF_SEQUENCE = [0, 0, 1]

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
    self_structure = get_tree_config("depth_2_tree_config")
    forcing_tree = FEX(
        sample_indices=SELF_SEQUENCE,
        leaf_dim=3,
        num_leaves=self_structure.num_leaves,
        tree_structure=self_structure,
    ).to(device)
    initialize_small(forcing_tree)

    config = FEXConfig(
        num_epochs=args.epochs,
        bfgs_epochs=args.bfgs_epochs,
        lr=args.lr,
        bfgs_lr=0.5,
        leaf_dim=3,
        num_leaves=self_structure.num_leaves,

        target_dim=args.dim,
    )

    print(f"Device: {device}")
    print(f"Self operator sequence: {SELF_SEQUENCE}")
    print(
        f"Fine-tuning: Adam epochs={args.epochs}, BFGS iterations={args.bfgs_epochs}, "
        f"samples={args.samples}"
    )
    train_logger = runtimeconfig.CreateLogger(log_path=f"HR/logs/finetune_ground_truth_dim_{args.dim}.log")
    score = train_fex(
        forcing_tree,
        dataloader,
        config,
        device=device,
        verbose=True,
    )

    """recovered_self = self_coefficients(forcing_tree)
    target_self = self_targets()
    target_interaction = interaction_targets()
    self_error = maximum_error(recovered_self, target_self)
    coefficient_error = self_error"""

    mse = evaluate_mse(forcing_tree, dataloader, device, args.dim)


    print(f"\nReturned score: {score:.8e}")
    print(f"Evaluation MSE: {mse:.8e}")

    print(f"Cleaned SymPy threshold: {args.expression_threshold:g}")
    # print("Self SymPy:", clean_self)
    # print("Interaction SymPy:", clean_interaction)
    # print(f"Maximum self coefficient error: {self_error:.8e}")
    # print(f"Maximum interaction coefficient error: {interaction_error:.8e}")

    passed = (
        math.isfinite(mse)
        and mse <= args.mse_tolerance
        # and coefficient_error <= args.coefficient_tolerance
    )
    print("RESULT:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
