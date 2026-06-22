"""Fine-tune the known HR operator structures and verify recovered coefficients."""

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
from FEX.training.train_configs import FEXConfig
from FEX.training.train_fex import train_network_fex
from FEX.training.tree_helpers import apply_inter_leaf_masks, traverse
from FEX.utils.tree_configs import get_tree_config


SELF_OP_SEQUENCE = [0, 0, 0, 0, 1, 2, 0]
INTERACTION_OP_SEQUENCE = [1, 0, 3]

SELF_LEAF_DIMS = [1, 0, 0, 2]  # y_i, x_i, x_i, z_i
INTERACTION_LEAF_DIMS = [0, 3]  # x_i, x_j

SELF_TARGET = {
    "y": 1.0,
    "x_squared": 3.0,
    "x_cubed": -1.0,
    "z": -1.0,
    "constant": 3.24,
}
INTERACTION_TARGET = {
    "x_sigmoid": -0.15,
    "x": 0.0,
    "sigmoid": 0.30,
    "constant": 0.0,
}


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_adjacency(num_nodes: int, edge_probability: float, device) -> torch.Tensor:
    adjacency = (torch.rand(num_nodes, num_nodes) < edge_probability).float()
    adjacency.fill_diagonal_(0.0)

    # Keep every node represented in the interaction loss.
    for node in range(num_nodes):
        if adjacency[node].sum() == 0:
            adjacency[node, (node + 1) % num_nodes] = 1.0

    return adjacency.to(device)


def make_synthetic_hr_data(
    num_samples: int,
    adjacency: torch.Tensor,
    device,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_nodes = adjacency.size(0)
    x = torch.empty(num_samples, num_nodes, 3, device=device)
    x[..., 0].uniform_(-2.0, 2.0)
    x[..., 1].uniform_(-8.0, 4.0)
    x[..., 2].uniform_(0.0, 5.0)

    x_i = x[..., 0]
    y_i = x[..., 1]
    z_i = x[..., 2]
    self_dynamics = y_i - x_i.pow(3) + 3.0 * x_i.pow(2) - z_i + 3.24

    sigmoid_neighbor = torch.sigmoid(x_i)
    interaction = 0.15 * (2.0 - x_i).unsqueeze(2) * sigmoid_neighbor.unsqueeze(1)
    interaction = interaction * adjacency.unsqueeze(0)
    dx = self_dynamics + interaction.sum(dim=2)

    derivatives = torch.zeros_like(x)
    derivatives[..., 0] = dx
    return x.cpu(), derivatives.cpu()


def set_hard_leaf_selection(fex: FEX, selected_dims: list[int]) -> None:
    if len(fex.leaf_mlps) != len(selected_dims):
        raise ValueError("Leaf selection count does not match the FEX tree.")

    with torch.no_grad():
        for leaf, selected_dim in zip(fex.leaf_mlps, selected_dims):
            leaf.logits.fill_(-20.0)
            leaf.logits[selected_dim] = 20.0
            leaf.logits.requires_grad_(False)


def unary_parameters(fex: FEX) -> list[tuple[str, float, float]]:
    parameters = []

    def collect(node):
        if node.operation_type == "unary":
            parameters.append(
                (
                    node.operation.op.__name__,
                    float(node.operation.a.detach().item()),
                    float(node.operation.b.detach().item()),
                )
            )

    traverse(fex.parent_node, collect)
    return parameters


def effective_self_coefficients(forcing_tree: FEX) -> dict[str, float]:
    params = unary_parameters(forcing_tree)
    if len(params) != 4:
        raise ValueError(f"Expected four self unary nodes, found {len(params)}.")

    return {
        "y": params[0][1],
        "x_squared": params[1][1],
        "x_cubed": params[2][1],
        "z": params[3][1],
        "constant": sum(param[2] for param in params),
    }


def effective_interaction_coefficients(interaction_tree: FEX) -> dict[str, float]:
    params = unary_parameters(interaction_tree)
    if len(params) != 2:
        raise ValueError(f"Expected two interaction unary nodes, found {len(params)}.")

    _, left_a, left_b = params[0]
    _, right_a, right_b = params[1]
    return {
        "x_sigmoid": left_a * right_a,
        "x": left_a * right_b,
        "sigmoid": left_b * right_a,
        "constant": left_b * right_b,
    }


def maximum_error(actual: dict[str, float], target: dict[str, float]) -> float:
    return max(abs(actual[key] - target[key]) for key in target)


def evaluate_mse(forcing_tree, interaction_tree, dataloader, adjacency, device) -> float:
    forcing_tree.eval()
    interaction_tree.eval()
    nodes, edges = adjacency.nonzero(as_tuple=True)
    non_self = nodes != edges
    nodes = nodes[non_self].to(device)
    edges = edges[non_self].to(device)

    squared_error = 0.0
    element_count = 0
    with torch.no_grad():
        for batch_x, batch_dy in dataloader:
            batch_x = batch_x.to(device)
            target = batch_dy[..., 0:1].to(device)
            batch_size, num_nodes, _ = batch_x.shape

            self_out = forcing_tree(batch_x.reshape(batch_size * num_nodes, -1))
            self_out = self_out.reshape(batch_size, num_nodes, 1)

            edge_inputs = torch.cat(
                [batch_x[:, nodes, :], batch_x[:, edges, :]],
                dim=-1,
            )
            inter_out = interaction_tree(edge_inputs.reshape(-1, edge_inputs.size(-1)))
            inter_out = inter_out.reshape(batch_size, nodes.numel(), 1)

            predicted_interaction = torch.zeros_like(self_out)
            scatter_index = nodes.view(1, -1, 1).expand(batch_size, -1, 1)
            predicted_interaction.scatter_add_(1, scatter_index, inter_out)
            prediction = self_out + predicted_interaction

            squared_error += torch.square(prediction - target).sum().item()
            element_count += target.numel()

    return squared_error / max(element_count, 1)


def print_comparison(title, actual, target) -> None:
    print(f"\n{title}")
    print(f"{'coefficient':<14} {'recovered':>12} {'target':>12} {'abs error':>12}")
    for key, target_value in target.items():
        recovered = actual[key]
        print(f"{key:<14} {recovered:>12.6f} {target_value:>12.6f} {abs(recovered - target_value):>12.6f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--bfgs-epochs", type=int, default=50)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--nodes", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--coefficient-tolerance", type=float, default=0.05)
    parser.add_argument("--mse-tolerance", type=float, default=1e-4)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adjacency = make_adjacency(args.nodes, edge_probability=0.35, device=device)
    train_x, train_dy = make_synthetic_hr_data(args.samples, adjacency, device)
    dataloader = DataLoader(
        TensorDataset(train_x, train_dy),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )

    forcing_config = get_tree_config("depth_3_leaves_4_config")
    interaction_config = get_tree_config("depth_2_tree_config")
    forcing_tree = FEX(
        sample_indices=torch.tensor(SELF_OP_SEQUENCE, dtype=torch.long, device=device),
        leaf_dim=3,
        num_leaves=forcing_config.num_leaves,
        tree_structure=forcing_config,
    ).to(device)
    interaction_tree = FEX(
        sample_indices=torch.tensor(INTERACTION_OP_SEQUENCE, dtype=torch.long, device=device),
        leaf_dim=6,
        num_leaves=interaction_config.num_leaves,
        tree_structure=interaction_config,
    ).to(device)

    apply_inter_leaf_masks(interaction_tree, node_dim=3)
    set_hard_leaf_selection(forcing_tree, SELF_LEAF_DIMS)
    set_hard_leaf_selection(interaction_tree, INTERACTION_LEAF_DIMS)

    config = FEXConfig(
        num_epochs=args.epochs,
        bfgs_epochs=args.bfgs_epochs,
        lr=args.lr,
        inter_lr=args.lr,
        bfgs_lr=0.5,
        leaf_dim=3,
        num_leaves=forcing_config.num_leaves,
        tau_start=1.0,
        tau_end=1.0,
        tau_schedule="non_decay",
    )

    print(f"Device: {device}")
    print(f"Self operator sequence: {SELF_OP_SEQUENCE}")
    print(f"Interaction operator sequence: {INTERACTION_OP_SEQUENCE}")
    score = train_network_fex(
        forcing_tree,
        interaction_tree,
        dataloader,
        adjacency,
        config,
        device=device,
    )

    self_coefficients = effective_self_coefficients(forcing_tree)
    interaction_coefficients = effective_interaction_coefficients(interaction_tree)
    mse = evaluate_mse(forcing_tree, interaction_tree, dataloader, adjacency, device)

    print_comparison("Self dynamics", self_coefficients, SELF_TARGET)
    print_comparison("Interaction dynamics", interaction_coefficients, INTERACTION_TARGET)
    print(f"\nReturned training score: {score:.8g}")
    print(f"Evaluation MSE: {mse:.8g}")
    print("Self expression:", forcing_tree.simplified_expression(["x_i", "y_i", "z_i"]))
    print(
        "Interaction expression:",
        interaction_tree.simplified_expression(
            ["x_i", "y_i", "z_i", "x_j", "y_j", "z_j"]
        ),
    )

    coefficient_error = max(
        maximum_error(self_coefficients, SELF_TARGET),
        maximum_error(interaction_coefficients, INTERACTION_TARGET),
    )
    passed = (
        math.isfinite(mse)
        and mse <= args.mse_tolerance
        and coefficient_error <= args.coefficient_tolerance
    )
    print(f"Maximum effective coefficient error: {coefficient_error:.8g}")
    print("RESULT:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
