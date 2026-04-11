
__all__ = ["FEX"]

import torch
import torch.nn as nn
import torch.nn.functional as F
from .nodes import Node
from ..utils.tree_configs import TREE_CONFIGS, get_tree_config

import logging
tree_logger = logging.getLogger("debug_tree")

from ..training.tree_helpers import traverse, fex_state_dict, fex_load_state_dict

class LeafMLP(nn.Module):
    def __init__(self, input_dim, init_tau=1.0, hard=False):
        super().__init__()
        self.logits = nn.Parameter(torch.randn(input_dim) * 0.1)
        self.register_buffer("tau", torch.tensor(init_tau, dtype=torch.float32))

        # self.sigma = nn.Parameter(torch.tensor(0.0))

        self.hard = hard
    
    def forward(self, leaf_input: torch.Tensor):
        logits = self.logits
        # logits = (logits - logits.mean()) / (logits.std() + 1e-8)
        weights = F.gumbel_softmax(logits, dim=-1, hard=self.hard, tau=float(self.tau.item()))
        out = (weights * leaf_input).sum(dim=-1, keepdim=True)
        # out = out + leaf_input.sum(dim=-1, keepdim=True) - leaf_input.detach().sum(dim=-1, keepdim=True)

        # print(f"Weights: {weights.detach().cpu().numpy()}")
        return out

    def set_hard(self, hard: bool):
        self.hard = hard

    def set_tau(self, tau: float):
        self.tau.fill_(max(float(tau), 1e-3))

    def get_tau(self):
        return float(self.tau.item())
    
    def selected_dim(self):
        return int(self.logits.argmax().item())

    def selection_confidence(self):
        probs = torch.softmax(self.logits.detach(), dim=-1)
        return float(probs.max().item())

    def expression(self):
        selected_dim = self.selected_dim()
        confidence = self.selection_confidence()
        # scale = float(self.scaling_param[selected_dim].detach().item())
        # bias = float(self.bias_param.detach().item())
        # bias_sign = "+" if bias >= 0 else "-"
        return f"x[{selected_dim}] (conf: {confidence:.2f})" # {bias_sign} {abs(bias):.3f}"
    

class FEX(nn.Module):
    def __init__(self, leaf_dim, num_leaves, sample_indices=None, tree_structure=None, parent_node=None, init_tau=5.0): 
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

        leaf_mlps = [LeafMLP(self.leaf_dim, init_tau) for _ in range(num_leaves)]
        for idx, leaf in enumerate(leaf_mlps):
            leaf._debug_leaf_idx = idx
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

    def leaf_params(self):
        """Parameters controlling leaf dimension selection (logits + sigma)."""
        for leaf_mlp in self.leaf_mlps:
            yield from leaf_mlp.parameters()

    def tree_params(self):
        """Parameters controlling tree node scalars (sign, magnitude, bias)."""
        return list(self.parent_node.get_parameters())
    
    def tree_log_mags(self):
        """Helper to extract log magnitudes from tree nodes for regularization."""
        log_mags = []
        def action(node: Node):
            if node.operation_type == "unary":
                log_mags.append(node.operation.log_mag)
        traverse(self.parent_node, action)
        return log_mags

    def to(self, device):
        super().to(device)
        self.parent_node.to(device)
        return self
    
    def freeze_b(self):
        def action(node: Node):
            if node.operation_type == "unary":
                node.operation.b.requires_grad = False
        traverse(self.parent_node, action)

    def unfreeze_b(self):
        def action(node: Node):
            if node.operation_type == "unary":
                node.operation.b.requires_grad = True
        traverse(self.parent_node, action)


    """ Overload train/eval to propogate to tree nodes"""
    def _set_tree_training_mode(self, mode: bool):
        def action(node: Node):
            if node.operation_type in ["unary", "binary"]:
                node.operation.train(mode)
        traverse(self.parent_node, action)

    def train(self, mode: bool = True):
        super().train(mode)
        self._set_tree_training_mode(mode)
        return self

    """ Gumbel Softmax Control """
    def set_leaf_tau(self, tau: float):
        for leaf_mlp in self.leaf_mlps:
            leaf_mlp.set_tau(tau)

    def get_leaf_tau(self):
        if len(self.leaf_mlps) == 0:
            return None
        return self.leaf_mlps[0].get_tau()
    
    def set_leaf_hard(self, hard: bool):
        for leaf_mlp in self.leaf_mlps:
            leaf_mlp.set_hard(hard)
    

    # _tree_config_name: Helper to identify which tree config was used for this FEX instance
    def _tree_config_name(self):
        if self.tree_structure is None:
            return None
        for name, config in TREE_CONFIGS.items():
            if config is self.tree_structure:
                return name
        return None

    def state_dict(self, *args, **kwargs):
        return fex_state_dict(self, *args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        return fex_load_state_dict(self, state_dict, strict=strict)
    
    # Deep copy of FEX for saving best candidates during score computation (T1/T2)
    def copy_inorder(self):
        copied_parent = self.parent_node.copy_inorder()
        copied_fex = FEX(
            leaf_dim=self.leaf_dim,
            num_leaves=len(self.leaf_mlps),
            parent_node=copied_parent,
            sample_indices=self.sample_indices,
            tree_structure=self.tree_structure,
        )
        # Copy leaf MLP parameters
        for copied_leaf_mlp, original_leaf_mlp in zip(copied_fex.leaf_mlps, self.leaf_mlps):
            copied_leaf_mlp.load_state_dict(original_leaf_mlp.state_dict())
        return copied_fex

    
    def __str__(self):
        leaf_expressions = [leaf_mlp.expression() for leaf_mlp in self.leaf_mlps]
        return self.parent_node.__str__(leaf_expressions=leaf_expressions)
    









"""  Debug FEX Tree  """
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