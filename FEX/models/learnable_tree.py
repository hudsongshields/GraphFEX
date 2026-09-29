
__all__ = ["FEX"]

import math
from statistics import mode

import torch
import torch.nn as nn
from .nodes import Node, UnaryOperation, BinaryOperation
from ..helpers.tree_configs import TREE_CONFIGS

import logging
tree_logger = logging.getLogger("debug_tree")

from ..training.tree_helpers import traverse, fex_state_dict, fex_load_state_dict

class LeafMLP(nn.Module):
    def __init__(self, input_dim, **kwargs):
        super().__init__()
        weight = torch.empty(1, input_dim)
        nn.init.xavier_uniform_(weight)
        self.logits = nn.Parameter(weight.squeeze(0))
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, leaf_input: torch.Tensor):
        return torch.sum(leaf_input * self.logits, dim=-1, keepdim=True) + self.bias

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
        
        self.leaf_mlps = nn.ModuleList()
        self.unary_ops = nn.ModuleList()
        self.binary_ops = nn.ModuleList()

        if parent_node:
            self.parent_node = parent_node
        elif tree_structure is not None and sample_indices is not None:
            raw_tree = tree_structure.build_tree(sample_indices)
            self.parent_node = self._register_tree(raw_tree)
        else:
            self.parent_node = None

        self.expr_thresh = kwargs.get("expression_threshold", 0.001)

    def _register_tree(self, raw_node) -> Node:
        if raw_node is None:
            return None

        op_type = raw_node.operation_type
        op_idx = None

        if op_type == "leaf" or (raw_node.left is None and raw_node.right is None):
            op_type = "leaf"
            while len(self.leaf_mlps) <= raw_node.leaf_idx:
                self.leaf_mlps.append(LeafMLP(self.leaf_dim))
        
        elif op_type == "unary":
            op_idx = len(self.unary_ops)
            self.unary_ops.append(UnaryOperation(raw_node.operation))
            
        elif op_type == "binary":
            op_idx = len(self.binary_ops)
            self.binary_ops.append(BinaryOperation(raw_node.operation))

        left_node = self._register_tree(raw_node.left)
        right_node = self._register_tree(raw_node.right)

        return Node(
            operation_type=op_type,
            operation_idx=op_idx,
            leaf_idx=raw_node.leaf_idx,
            left=left_node,
            right=right_node,
            name=raw_node.name
        )

    # @torch.compile
    def forward(self, x: torch.Tensor):
        def compute_node(node: Node):
            if node.operation_type == "leaf":
                return self.leaf_mlps[node.leaf_idx](x)

            elif node.operation_type == "unary":
                child = compute_node(node.left)
                return self.unary_ops[node.operation_idx](child)

            elif node.operation_type == "binary":
                left_val = compute_node(node.left)
                right_val = compute_node(node.right)
                return self.binary_ops[node.operation_idx](left_val, right_val)

            raise ValueError("Invalid node layout metadata discovered.")

        return compute_node(self.parent_node)

    def to(self, device):
        super().to(device)
        self.parent_node.to(device)
        return self
    


    """ Overload train/eval to propogate to tree nodes"""
    def train(self, mode: bool = True):
        return super().train(mode)


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

    def expression_summary(self):
        leaf_expressions = [str(leaf) for leaf in self.leaf_mlps]

        def build(node):
            if node.operation_type == "leaf":
                return f"({leaf_expressions[node.leaf_idx]})"

            elif node.operation_type == "unary":
                op = self.unary_ops[node.operation_idx]

                a = op.a.detach().item()
                b = op.b.detach().item()

                return (
                    f"({a:.3f} * "
                    f"{op.op.__name__}({build(node.left)}) "
                    f"+ {b:.3f})"
                )

            elif node.operation_type == "binary":
                op = self.binary_ops[node.operation_idx]

                return (
                    f"({build(node.left)} "
                    f"{op.op.__name__} "
                    f"{build(node.right)})"
                )

            raise ValueError(
                f"Unknown operation type: {node.operation_type}"
            )

        return build(self.parent_node)

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
            if op_name == "sin":
                return sp.sin(value)    
            raise ValueError(f"Unsupported unary operator: {op_name}")

        def build_leaf(leaf_idx):
            leaf = self.leaf_mlps[leaf_idx]
            expression = rounded_parameter(leaf.bias)
            for coefficient, symbol in zip(leaf.logits, symbols):
                expression += rounded_parameter(coefficient) * symbol
            return expression
        
        def build(node):
            if node.operation_type == "leaf":
                return build_leaf(node.leaf_idx)

            if node.operation_type == "binary":
                left = build(node.left)
                right = build(node.right)

                op_name = self.binary_ops[node.operation_idx].op.__name__

                if op_name == "add":
                    return left + right
                if op_name == "sub":
                    return left - right
                if op_name == "mul":
                    return left * right
                if op_name == "safe_div":
                    return left / right

                raise ValueError(f"Unsupported binary operator: {op_name}")

            if node.operation_type == "unary":
                child = build(node.left)
                op_name = self.unary_ops[node.operation_idx].op.__name__

                transformed = apply_unary(op_name, child)

                return (rounded_parameter(self.unary_ops[node.operation_idx].a) * transformed + rounded_parameter(self.unary_ops[node.operation_idx].b))

        expanded = sp.expand(build(self.parent_node))
        retained_terms = []
        for term in sp.Add.make_args(expanded):
            coefficient, _ = term.as_coeff_Mul()
            if not coefficient.is_number or abs(float(coefficient)) >= self.expr_thresh:
                retained_terms.append(term)

        return sp.simplify(sp.Add(*retained_terms))

    def simplified_expression(self, variable_names=None):
        return str(self.symbolic_expression(variable_names))
    