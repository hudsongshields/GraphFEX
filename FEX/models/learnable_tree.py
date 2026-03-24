from argparse import __all__
__all__ = ["FEX"]

import torch
import torch.nn as nn
from .nodes import Node
from ..utils.tree_configs import TREE_CONFIGS, get_tree_config

import logging
tree_logger = logging.getLogger("debug_tree")

class LeafMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x: torch.Tensor):
        return self.linear(x)

    def expression(self):
        weights = self.linear.weight.detach().flatten().tolist()
        bias = float(self.linear.bias.detach().item())

        terms = []
        for i, w in enumerate(weights):
            sign = "+" if w >= 0 else "-"
            terms.append(f"{sign} {abs(w):.3f}*X_{i}")

        bias_sign = "+" if bias >= 0 else "-"
        return " ".join(terms) + f" {bias_sign} {abs(bias):.3f}"
    

class FEX(nn.Module):
    def __init__(self, leaf_dim, num_leaves, sample_indices=None, tree_structure=None, parent_node=None): 
        super().__init__()
        self.leaf_dim = leaf_dim
        
        self.sample_indices = sample_indices
        self.tree_structure = tree_structure
        if parent_node:
            self.parent_node = parent_node
        elif tree_structure is not None and sample_indices is not None:
            self.parent_node = tree_structure.build_tree(sample_indices)
        else:
            self.parent_node = None

        leaf_mlps = [LeafMLP(self.leaf_dim) for _ in range(num_leaves)]
        self.leaf_mlps = nn.ModuleList(leaf_mlps)


    def forward(self, x: torch.Tensor):
        leaf_outputs = [leaf_mlp(x) for leaf_mlp in self.leaf_mlps]
        
        def compute_node(node: Node, depth: int = 0):
            indent = "  " * depth
            tree_logger.debug(f"{indent}Entering {node.operation_type}")

            if node.operation_type == "leaf":
                out = leaf_outputs[node.leaf_idx]

            elif node.operation_type == "unary":
                child = compute_node(node.left, depth + 1)
                out = node.operation(child)

            elif node.operation_type == "binary":
                left_val = compute_node(node.left, depth + 1)
                right_val = compute_node(node.right, depth + 1)
                out = node.operation(left_val, right_val)

            tree_logger.debug(f"{indent}Returning from {node.operation_type} -> shape {out.shape}")
            return out

        return compute_node(self.parent_node)
    
    def all_parameters(self):
        yield from self.parameters()
        yield from self.parent_node.get_parameters()

    def tree_params(self):
        return list(self.parent_node.get_parameters())

    def to(self, device):
        super().to(device)
        self.parent_node.to(device)
        return self
    
    def visualize_tree(self, filename=None, format="png"):
        leaf_transforms=[]
        for leaf_idx, leaf_mlp in enumerate(self.leaf_mlps):
            mlp_str = f"leaf{leaf_idx}: {leaf_mlp.expression()}"
            leaf_transforms.append(mlp_str)

        return self.parent_node.visualize_tree_inorder(
            filename=filename,
            format=format,
            leaf_transforms=leaf_transforms,
        )

    """
    Operations to enable saving/loading entire FEX model
    _tree_config_name: Helper to identify which tree config was used for this FEX instance
    state_dict: Override to include tree structure and sample indices in the checkpoint
    load_state_dict: Override to reconstruct tree structure and sample indices when loading from checkpoint
    """
    def _tree_config_name(self):
        if self.tree_structure is None:
            return None
        for name, config in TREE_CONFIGS.items():
            if config is self.tree_structure:
                return name
        return None

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        tree_name = self._tree_config_name()
        if tree_name is not None:
            state["_meta_tree_name"] = tree_name

        if self.sample_indices is not None:
            if isinstance(self.sample_indices, torch.Tensor):
                sample_indices = self.sample_indices.detach().cpu().to(dtype=torch.int64).tolist()
            else:
                sample_indices = [int(i) for i in self.sample_indices]
            state["_meta_sample_indices"] = sample_indices

        state["_meta_leaf_dim"] = int(self.leaf_dim)
        state["_meta_num_leaves"] = int(len(self.leaf_mlps))

        if self.parent_node is not None:
            state.update(self.parent_node.state_dict(prefix="tree."))
        return state
    

    def load_state_dict(self, checkpoint, strict=True):
        state = dict(checkpoint["state_dict"])

        tree_name = state.pop("_meta_tree_name")
        sample_indices = [int(i) for i in state.pop("_meta_sample_indices")]
        state.pop("_meta_leaf_dim", None)
        state.pop("_meta_num_leaves", None)

        self.tree_structure = get_tree_config(tree_name)
        self.sample_indices = sample_indices
        self.parent_node = self.tree_structure.build_tree(sample_indices)

        tree_state = {k: v for k, v in state.items() if k.startswith("tree.")}
        module_state = {k: v for k, v in state.items() if not k.startswith("tree.")}

        result = super().load_state_dict(module_state, strict=strict)
        self.parent_node.load_state_dict(tree_state, prefix="tree.")

        return result
    
    
    # Deep copy of FEX for saving best candidates during score computation (T1/T2)
    def copy_inorder(self):
        copied_parent = self.parent_node.copy_inorder()
        copied_fex = FEX(leaf_dim=self.leaf_dim, num_leaves=len(self.leaf_mlps), parent_node=copied_parent)
        # Copy leaf MLP parameters
        for copied_leaf_mlp, original_leaf_mlp in zip(copied_fex.leaf_mlps, self.leaf_mlps):
            copied_leaf_mlp.load_state_dict(original_leaf_mlp.state_dict())
        return copied_fex


    def __str__(self):
        leaf_expressions = [leaf_mlp.expression() for leaf_mlp in self.leaf_mlps]
        return self.parent_node.__str__(leaf_expressions=leaf_expressions)
    


# ------------ Debug FEX Tree ------------- #
if __name__ == "__main__":
    from .controllers import Controller
    from ..utils.sampler import epsilon_greedy_sample
    from ..utils.operations import unary_operation, binary_operation, UNARY_OPS, BINARY_OPS

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("debug_tree").setLevel(logging.DEBUG) 

    NUM_NODES = 5
    """
    Example tree structure

    Node 0: Binary (Node 1, Node 2)
    Node 1: Unary (Node 3)
    Node 2: Unary (Node 4)
    Node 3: Leaf
    Node 4: Leaf
    """

    ops_per_node = [BINARY_OPS, UNARY_OPS, UNARY_OPS]
    sample_indices = [1, 2, 3]
    leaf1 = Node(operation_type="leaf", leaf_idx=0, name="leaf1", operation=None)
    leaf2 = Node(operation_type="leaf", leaf_idx=1, name="leaf2", operation=None)
    branch1 = Node(operation_type="unary", operation=unary_operation(sample_indices[1]), left=leaf1, name="branch1")
    branch2 = Node(operation_type="unary", operation=unary_operation(sample_indices[2]), left=leaf2, name="branch2")
    parent_node = Node(operation_type="binary", operation=binary_operation(sample_indices[0]), left=branch1, right=branch2, name="parent_node")

    learnable_tree = FEX(leaf_dim=10, num_leaves=2, parent_node=parent_node)
    fake_x = torch.randn(5, 10)
    output = learnable_tree(fake_x)
    print(output)