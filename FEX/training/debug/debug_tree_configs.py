import torch

from ...models.learnable_tree import FEX
from ...utils.tree_configs import get_tree_config

def set_mag(node, mag, sign):
    with torch.no_grad():
        node.a.fill_(mag * sign)


def set_hard_leaf(leaf_mlp, dim_idx: int, logit_value: float = 10000.0):
    with torch.no_grad():
        leaf_mlp.logits.fill_(-logit_value)
        leaf_mlp.logits[dim_idx] = logit_value


def build_debug_dx_forcing_fex(node_dim: int, device: str) -> FEX:

    forcing_config = get_tree_config("depth_3_leaves_4_config")
    sample_indices = torch.tensor([0, 0, 0, 0, 2, 1, 0], dtype=torch.long, device=device)
    model = FEX(
        sample_indices=sample_indices,
        leaf_dim=node_dim,
        num_leaves=forcing_config.num_leaves,
        tree_structure=forcing_config,
        init_tau=0.01,
        epsilon_greedy=0.0
    ).to(device)

    with torch.no_grad():
        # term1: x_i2 + 3.2392
        set_mag(model.parent_node.left.left.operation, 1.0, 1)
        model.parent_node.left.left.operation.b.fill_(3.24)

        # term2: -x_i1^3
        set_mag(model.parent_node.left.right.operation, 1.0, -1) # change to the wrong sign to test if can recover
        model.parent_node.left.right.operation.b.fill_(0.0)

        # term3: +3.0*x_i1^2
        set_mag(model.parent_node.right.left.operation, 3.0, 1)
        model.parent_node.right.left.operation.b.fill_(0.0)

        # term4: -x_i3
        set_mag(model.parent_node.right.right.operation, 1.0, -1)
        model.parent_node.right.right.operation.b.fill_(0.0)

        set_hard_leaf(model.leaf_mlps[0], 1)
        set_hard_leaf(model.leaf_mlps[1], 0)
        set_hard_leaf(model.leaf_mlps[2], 0)
        set_hard_leaf(model.leaf_mlps[3], 2)

    return model


def build_debug_dy_forcing_fex(node_dim: int, device: str) -> FEX:
    """
    dy_i/dt = 1.0004 - 5.0001*x_i1^2 - 1.0001*x_i2
    """
    partial_config = get_tree_config("depth_3_partial_config")
    sample_indices = torch.tensor([2, 2, 0, 0, 1], dtype=torch.long, device=device)
    model = FEX(
        sample_indices=sample_indices,
        leaf_dim=node_dim,
        num_leaves=partial_config.num_leaves,
        tree_structure=partial_config,
        init_tau=0.01,
    ).to(device)

    with torch.no_grad():
        # left.left: constant 1.0004 (a=0, so log_mag=0)
        set_mag(model.parent_node.left.left.operation, 0.0, 1)
        model.parent_node.left.left.operation.b.fill_(1.0004)

        # left.right: 5.0001*x_i1^2
        set_mag(model.parent_node.left.right.operation, 5.0001, 1)
        model.parent_node.left.right.operation.b.fill_(0.0)

        # right: +1.0001*x_i2
        set_mag(model.parent_node.right.operation, 1.0001, 1)
        model.parent_node.right.operation.b.fill_(0.0)

        set_hard_leaf(model.leaf_mlps[0], 0)
        set_hard_leaf(model.leaf_mlps[1], 0)
        set_hard_leaf(model.leaf_mlps[2], 1)

    return model


def build_debug_dz_forcing_fex(node_dim: int, device: str) -> FEX:
    """
    dz_i/dt = r * [s * (x_i1 - x0) - x_i3]
    where r=0.004, s=4, x0=-1.6
    dz_i/dt = 0.016 * (x_i1 + 1.6) - 0.004 * x_i3
    """
    dz_config = get_tree_config("depth_2_tree_config")
    # sample_indices: [binary_op, unary_op_left, unary_op_right]
    # Use 0 for add, 0 for identity (left), 0 for identity (right) if UNARY_OPS[0] is identity
    sample_indices = torch.tensor([0, 0, 0], dtype=torch.long, device=device)
    model = FEX(
        sample_indices=sample_indices,
        leaf_dim=node_dim,
        num_leaves=dz_config.num_leaves,
        tree_structure=dz_config,
        init_tau=0.01,
    ).to(device)

    with torch.no_grad():
        # left: 0.016 * x1 + 0.0256
        set_mag(model.parent_node.left.operation, 0.016, 1)
        model.parent_node.left.operation.b.fill_(0.0256)

        # right: -0.004 * x3
        set_mag(model.parent_node.right.operation, 0.004, -1)
        model.parent_node.right.operation.b.fill_(0.0)

        set_hard_leaf(model.leaf_mlps[0], 0)  # x1
        set_hard_leaf(model.leaf_mlps[1], 2)  # x3

    return model


def build_debug_interaction_fex(node_dim: int, device: str) -> FEX:
    """
    g(x_i, x_j) = 0.3*sigmoid(x_j1) - 0.15*x_i1*sigmoid(x_j1)
    Input is concat([x_i1,x_i2,x_i3,x_j1,x_j2,x_j3]).
    """
    inter_config = get_tree_config("depth_2_tree_config")
    sample_indices = torch.tensor([1, 0, 5], dtype=torch.long, device=device)
    model = FEX(
        sample_indices=sample_indices,
        leaf_dim=node_dim * 2,
        num_leaves=inter_config.num_leaves,
        tree_structure=inter_config,
        init_tau=0.01,
        epsilon_greedy=0.0,
    ).to(device)

    with torch.no_grad():
        # left branch: -0.15 * x_i1 + 0.3
        set_mag(model.parent_node.left.operation, 0.15, -1)
        model.parent_node.left.operation.b.fill_(0.3)

        # right branch: 1.0 * sigmoid(x_j1) + 0
        set_mag(model.parent_node.right.operation, 1.0, 1)
        model.parent_node.right.operation.b.fill_(0.0)

        set_hard_leaf(model.leaf_mlps[0], 0)
        set_hard_leaf(model.leaf_mlps[1], 3)

    return model
