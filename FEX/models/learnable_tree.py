
__all__ = ["FEX"]

import math

import torch
import torch.nn as nn
from .nodes import Node
from ..utils.tree_configs import TREE_CONFIGS

import logging
tree_logger = logging.getLogger("debug_tree")

from ..training.tree_helpers import traverse, fex_state_dict, fex_load_state_dict

class LeafMLP(nn.Module):
    def __init__(self, input_dim, **kwargs):
        super().__init__()
        # One unrestricted coefficient per input dimension, matching the FEX paper.
        self.logits = nn.Parameter(torch.randn(input_dim) * 0.1)
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, leaf_input: torch.Tensor, unary_op=None):
        transformed = unary_op(leaf_input) if unary_op is not None else leaf_input
        return torch.sum(transformed * self.logits, dim=-1, keepdim=True) + self.bias

    def reset_parameters(self):
        with torch.no_grad():
            self.logits.normal_(mean=0.0, std=0.1)
            self.bias.zero_()

    
    """ Printable expression """
    def _selected_dim(self):
        return int(self.logits.detach().abs().argmax().item())

    def _selection_confidence(self):
        probs = torch.softmax(self.logits.detach().abs(), dim=-1)
        return float(probs.max().item())

    def __str__(self):
        terms = [
            f"{float(weight.detach()):.4f}*x[{index}]"
            for index, weight in enumerate(self.logits)
            if abs(float(weight.detach())) >= 1e-4
        ]
        terms.append(f"{float(self.bias.detach()):.4f}")
        return " + ".join(terms)
    

class FEX(nn.Module):
    def __init__(self, leaf_dim, sample_indices=None, tree_structure=None, parent_node=None, **kwargs): 
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

        leaf_mlps = [LeafMLP(self.leaf_dim) for _ in range(self.tree_structure.num_leaves)]
        for idx, leaf in enumerate(leaf_mlps):
            leaf._debug_leaf_idx = idx
        self.leaf_mlps = nn.ModuleList(leaf_mlps)
        self._normalize_leaf_unary_operations()

        self.expr_thresh = kwargs.get("expression_threshold", 0.001)


    def forward(self, x: torch.Tensor):
        def compute_node(node: Node, depth: int = 0):
            indent = "  " * depth
            tree_logger.debug(f"{indent}Entering {node.operation_type}")

            if node.operation_type == "leaf":
                out = self.leaf_mlps[node.leaf_idx](x)

            elif node.operation_type == "unary":
                if node.left.operation_type == "leaf":
                    out = self.leaf_mlps[node.left.leaf_idx](x, unary_op=node.operation.op)
                else:
                    child = compute_node(node.left, depth + 1)
                    out = node.operation(child)

            elif node.operation_type == "binary":
                left_val = compute_node(node.left, depth + 1)
                right_val = compute_node(node.right, depth + 1)
                out = node.operation(left_val, right_val)

            tree_logger.debug(f"{indent}Returning from {node.operation_type} -> shape {out.shape}")
            return out

        return compute_node(self.parent_node)

    def _normalize_leaf_unary_operations(self):
        """Leaf-level affine coefficients live in LeafMLP, not UnaryOperation."""
        def action(node: Node):
            if (
                node.operation_type == "unary"
                and node.left is not None
                and node.left.operation_type == "leaf"
            ):
                with torch.no_grad():
                    node.operation.a.fill_(1.0)
                    node.operation.b.zero_()
                node.operation.a.requires_grad_(False)
                node.operation.b.requires_grad_(False)

        traverse(self.parent_node, action)
    
    def all_parameters(self):
        yield from (parameter for parameter in self.parameters() if parameter.requires_grad)
        yield from (
            parameter
            for parameter in self.parent_node.get_parameters()
            if parameter.requires_grad
        )

    def reset(self, fex_config=None):
        for leaf in self.leaf_mlps:
            if hasattr(leaf, 'reset_parameters'):
                leaf.reset_parameters()

        # Reset all tree node parameters as in their constructor, but keep structure
        def action(node: Node):
            node.reset()
        traverse(self.parent_node, action)
        self._normalize_leaf_unary_operations()

    def leaf_params(self):
        """Parameters controlling leaf dimension selection (logits + sigma)."""
        for leaf_mlp in self.leaf_mlps:
            yield from leaf_mlp.parameters()

    def tree_params(self):
        """Parameters controlling tree node scalars (sign, magnitude, bias)."""
        return [
            parameter
            for parameter in self.parent_node.get_parameters()
            if parameter.requires_grad
        ]
    
    def tree_mags(self):
        """Return paper-style leaf scaling vectors for magnitude regularization."""
        return [leaf.logits for leaf in self.leaf_mlps]

    def to(self, device):
        super().to(device)
        self.parent_node.to(device)
        return self
    


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


    # helper to identify which tree config was used for this FEX instance
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
        copied_sample_indices = self.sample_indices
        if isinstance(copied_sample_indices, torch.Tensor):
            copied_sample_indices = copied_sample_indices.detach().clone().cpu()
        copied_fex = FEX(
            leaf_dim=self.leaf_dim,
            num_leaves=len(self.leaf_mlps),
            parent_node=copied_parent,
            sample_indices=copied_sample_indices,
            tree_structure=self.tree_structure,
        )
        # Copy leaf MLP parameters
        for copied_leaf_mlp, original_leaf_mlp in zip(copied_fex.leaf_mlps, self.leaf_mlps):
            copied_leaf_mlp.load_state_dict(original_leaf_mlp.state_dict())
        return copied_fex

    
    def __str__(self):
        return self.simplified_expression()

    def symbolic_expression(self, variable_names=None):
        """Build the paper-style elementwise leaf expression with SymPy."""
        import sympy as sp
        digits = -math.floor(math.log10(self.expr_thresh)) + 1

        if variable_names is None:
            variable_names = [f"x{i + 1}" for i in range(self.leaf_dim)]
        if len(variable_names) != self.leaf_dim:
            raise ValueError(
                f"Expected {self.leaf_dim} variable names, got {len(variable_names)}."
            )

        symbols = sp.symbols(" ".join(variable_names))
        if self.leaf_dim == 1:
            symbols = (symbols,)

        def rounded_parameter(value):
            number = round(float(value.detach().item()), digits)
            return sp.Float(0.0 if abs(number) < self.expr_thresh else number)

        def apply_unary(op_name, value):
            if op_name == "identity":
                return value
            if op_name == "square":
                return value ** 2
            if op_name == "cube":
                return value ** 3
            if op_name == "fourth_power":
                return value ** 4
            if op_name == "safe_exp":
                return sp.exp(value)
            if op_name == "sigmoid":
                return 1 / (1 + sp.exp(-value))
            if op_name == "safe_reciprocal":
                return 1 / value
            raise ValueError(f"Unsupported unary operator: {op_name}")

        def build_leaf(leaf_idx, op_name="identity"):
            leaf = self.leaf_mlps[leaf_idx]
            threshold = rounded_parameter(leaf.bias)
            for coefficient, symbol in zip(leaf.logits, symbols):
                threshold += rounded_parameter(coefficient) * apply_unary(op_name, symbol)
            return threshold

        def build(node):
            if node.operation_type == "leaf":
                return build_leaf(node.leaf_idx)

            if node.operation_type == "binary":
                left = build(node.left)
                right = build(node.right)
                op_name = node.operation.op.__name__
                if op_name == "add":
                    return left + right
                if op_name == "sub":
                    return left - right
                if op_name == "mul":
                    return left * right
                if op_name == "safe_div":
                    return left / right
                raise ValueError(f"Unsupported binary operator: {op_name}")

            op_name = node.operation.op.__name__
            if node.left.operation_type == "leaf":
                return build_leaf(node.left.leaf_idx, op_name)

            child = build(node.left)
            transformed = apply_unary(op_name, child)
            return rounded_parameter(node.operation.a) * transformed + rounded_parameter(node.operation.b)

        expanded = sp.expand(build(self.parent_node))
        retained_terms = []
        for term in sp.Add.make_args(expanded):
            coefficient, _ = term.as_coeff_Mul()
            if not coefficient.is_number or abs(float(coefficient)) >= self.expr_thresh:
                retained_terms.append(term)

        return sp.simplify(sp.Add(*retained_terms))

    def simplified_expression(self, variable_names=None):
        return str(self.symbolic_expression(variable_names))
    
    """ external member functions """
    def visualize_tree(self, directory: str = "fex_tree_viz", clear_directory: bool = True):
        from ..training.tree_helpers import visualize_tree as vis
        return vis(self, filename=directory)
    









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
