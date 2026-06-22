from ..utils.tree_configs import get_tree_config
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models.nodes import Node
from graphviz import Digraph
import torch.nn.functional as F
import torch
import torch.nn as nn

from collections.abc import Callable
def traverse(node, action):
    if node is None:
        return
    action(node)
    if node.left is not None:
        traverse(node.left, action)
    if node.right is not None:
        traverse(node.right, action)


def apply_inter_leaf_masks(inter_fex, node_dim: int) -> None:
    """Constrain pairwise interaction leaves to source and neighbor feature blocks."""
    if inter_fex is None or len(getattr(inter_fex, "leaf_mlps", [])) < 2:
        return

    with torch.no_grad():
        if hasattr(inter_fex.leaf_mlps[0], "logit_mask"):
            inter_fex.leaf_mlps[0].logit_mask[node_dim:].fill_(-1e9)
        if hasattr(inter_fex.leaf_mlps[1], "logit_mask"):
            inter_fex.leaf_mlps[1].logit_mask[:node_dim].fill_(-1e9)


def copy_fex_state_(target, source) -> None:
    """Copy a trained FEX state into an existing FEX object in-place."""
    target.load_state_dict(source.state_dict(), strict=False)
    target.to(next(source.parameters()).device)


""" Debugging Helpers for FEX """
def get_noise_stds(fex) -> dict[str, torch.Tensor]:
    noise_stds = {}
    for leaf_idx, leaf_mlp in enumerate(fex.leaf_mlps):
        noise_stds[f"leaf{leaf_idx}"] = leaf_mlp.sigma
    return noise_stds
    

""" Storing Helpers for FEX """
def fex_state_dict(fex, *args, **kwargs):
    state = nn.Module.state_dict(fex, *args, **kwargs)
    tree_name = fex._tree_config_name()
    if tree_name is not None:
        state["_meta_tree_name"] = tree_name

    if fex.sample_indices is not None:
        if isinstance(fex.sample_indices, torch.Tensor):
            sample_indices = fex.sample_indices.detach().cpu().to(dtype=torch.int64).tolist()
        else:
            sample_indices = [int(i) for i in fex.sample_indices]
        state["_meta_sample_indices"] = sample_indices

    state["_meta_leaf_dim"] = int(fex.leaf_dim)
    state["_meta_num_leaves"] = int(len(fex.leaf_mlps))

    if fex.parent_node is not None:
        # Some Node variants do not implement state_dict/load_state_dict methods.
        # Only call those methods when present; otherwise use recursive helper.
        if hasattr(fex.parent_node, "state_dict"):
            try:
                tree_state = fex.parent_node.state_dict(prefix="tree.")
                if isinstance(tree_state, dict):
                    state.update(tree_state)
                else:
                    state.update(node_state_dict(fex.parent_node, prefix="tree."))
            except Exception:
                state.update(node_state_dict(fex.parent_node, prefix="tree."))
        else:
            state.update(node_state_dict(fex.parent_node, prefix="tree."))
    return state


def fex_load_state_dict(fex, checkpoint, strict=True):
    state = dict(checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint)

    tree_name = state.pop("_meta_tree_name", None)
    sample_indices = state.pop("_meta_sample_indices", None)
    if sample_indices is not None:
        sample_indices = [int(i) for i in sample_indices]
    state.pop("_meta_leaf_dim", None)
    state.pop("_meta_num_leaves", None)

    if tree_name is not None and sample_indices is not None:
        fex.tree_structure = get_tree_config(tree_name)
        fex.sample_indices = sample_indices
        fex.parent_node = fex.tree_structure.build_tree(sample_indices)

    tree_state = {k: v for k, v in state.items() if k.startswith("tree.")}
    module_state = {k: v for k, v in state.items() if not k.startswith("tree.")}

    result = nn.Module.load_state_dict(fex, module_state, strict=strict)
    if fex.parent_node is not None and len(tree_state) > 0:
        if hasattr(fex.parent_node, "load_state_dict"):
            try:
                fex.parent_node.load_state_dict(tree_state, prefix="tree.")
            except Exception:
                node_load_state_dict(fex.parent_node, tree_state, prefix="tree.")
        else:
            node_load_state_dict(fex.parent_node, tree_state, prefix="tree.")

    return result

""" Storing Helpers for FEX Nodes"""
def node_load_state_dict(node, state_dict, prefix=""):
    if isinstance(node.operation, nn.Module):
        op_prefix = f"{prefix}op."
        op_state = {
            key[len(op_prefix):]: value
            for key, value in state_dict.items()
            if key.startswith(op_prefix)
        }
        node.operation.load_state_dict(op_state, strict=False)

    if node.left is not None:
        node_load_state_dict(node.left, state_dict, prefix=f"{prefix}left.")

    if node.right is not None:
        node_load_state_dict(node.right, state_dict, prefix=f"{prefix}right.")

def node_state_dict(node, prefix=""):
    state = {}
    if isinstance(node.operation, nn.Module):
        for key, value in node.operation.state_dict().items():
            state[f"{prefix}op.{key}"] = value

    if node.left is not None:
        state.update(node_state_dict(node.left, prefix=f"{prefix}left."))

    if node.right is not None:
        state.update(node_state_dict(node.right, prefix=f"{prefix}right."))

    return state



def state_dict(target, *args, **kwargs):
    # Use string-based type hint for Node
    if "Node" in globals() and isinstance(target, globals()["Node"]):
        return node_state_dict(target, *args, **kwargs)
    return fex_state_dict(target, *args, **kwargs)



def load_state_dict(target, *args, **kwargs):
    # Use string-based type hint for Node
    if "Node" in globals() and isinstance(target, globals()["Node"]):
        return node_load_state_dict(target, *args, **kwargs)
    return fex_load_state_dict(target, *args, **kwargs)


""" Tree Visualization Helpers"""
def visualize_tree_inorder(node, filename=None, format="png", leaf_transforms=None):
    dot = Digraph()
    
    def safe_op_name(op):
        name = getattr(op, "__name__", str(op))
        # Graphviz interprets angle brackets as HTML labels.
        return name.replace("<", "(").replace(">", ")")

    def add_nodes_edges(node, parent_id=None):
        node_id = str(id(node))
        if node.operation_type == "leaf":
            if not leaf_transforms:
                label = f"Leaf {node.leaf_idx}"
            else:
                label = f"{leaf_transforms[node.leaf_idx]}"
        elif node.operation_type == "binary":
            label = f"binary: {safe_op_name(node.operation.op)}"
        elif node.operation_type == "unary":
            a, b = node.operation.a.item(), node.operation.b.item()
            label = f"{a:.3f} * {safe_op_name(node.operation.op)} + {b:.3f}"
        dot.node(node_id, label=label)

        if parent_id is not None:
            dot.edge(parent_id, node_id)

        if node.left:
            add_nodes_edges(node.left, node_id)
        if node.right:
            add_nodes_edges(node.right, node_id)

    add_nodes_edges(node)

    if filename is None:
        return dot

    dot.render(filename, format=format, cleanup=True)
    return dot


def visualize_tree(fex, filename=None, format="png"):
        leaf_transforms=[]
        for leaf_idx, leaf_mlp in enumerate(fex.leaf_mlps):
            mlp_str = f"leaf{leaf_idx}: {str(leaf_mlp)}"
            leaf_transforms.append(mlp_str)

        return visualize_tree_inorder(
            node=fex.parent_node,
            filename=filename,
            format=format,
            leaf_transforms=leaf_transforms,
        )
