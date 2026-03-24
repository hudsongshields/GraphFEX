import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Callable, Optional
import copy
from graphviz import Digraph


class UnaryOperation(nn.Module):
    def __init__(self, op: Callable):
        super().__init__()
        self.a = nn.Parameter(torch.tensor([1.0]))
        self.b = nn.Parameter(torch.tensor([0.0]))
        self.op = op

    def forward(self, x: torch.Tensor):
        return self.a * self.op(x) + self.b


class BinaryOperation(nn.Module):
    def __init__(self, op: Callable):
        super().__init__()
        self.op = op

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return self.op(x, y)


@dataclass
class Node:
    operation_type: str
    operation: Callable
    leaf_idx: Optional[int] = None
    left: Optional["Node"] = None
    right: Optional["Node"] = None

    name: Optional[str] = None

    def __post_init__(self):
        if self.left is None and self.right is None:
            self.operation_type = "leaf"

        if self.operation_type == "unary":
            if not isinstance(self.operation, UnaryOperation):
                self.operation = UnaryOperation(self.operation)
        elif self.operation_type == "binary":
            if not isinstance(self.operation, BinaryOperation):
                self.operation = BinaryOperation(self.operation)

    def preorder_traversal(self):
        nodes = [self]
        if self.left:
            nodes += self.left.preorder_traversal()
        if self.right:
            nodes += self.right.preorder_traversal()
        return nodes
    
    def get_parameters(self):
        params = []

        if isinstance(self.operation, nn.Module):
            params.extend(self.operation.parameters())

        if self.left is not None:
            params.extend(self.left.get_parameters())

        if self.right is not None:
            params.extend(self.right.get_parameters())

        return params
    
    def to(self, device):
        if isinstance(self.operation, nn.Module):
            self.operation.to(device)

        if self.left is not None:
            self.left.to(device)

        if self.right is not None:
            self.right.to(device)

        return self
    
    """Tools for storing and visualization"""
    def copy_inorder(self):
        return Node(
            operation_type=self.operation_type,
            operation=copy.deepcopy(self.operation),
            leaf_idx=self.leaf_idx,
            left=self.left.copy_inorder() if self.left else None,
            right=self.right.copy_inorder() if self.right else None,
            name=self.name
        )
    
    def load_state_dict(self, state_dict, prefix=""):
        if isinstance(self.operation, nn.Module):
            op_prefix = f"{prefix}op."
            op_state = {
                key[len(op_prefix):]: value
                for key, value in state_dict.items()
                if key.startswith(op_prefix)
            }
            self.operation.load_state_dict(op_state, strict=False)

        if self.left is not None:
            self.left.load_state_dict(state_dict, prefix=f"{prefix}left.")

        if self.right is not None:
            self.right.load_state_dict(state_dict, prefix=f"{prefix}right.")

    def state_dict(self, prefix=""):
        state = {}
        if isinstance(self.operation, nn.Module):
            for key, value in self.operation.state_dict().items():
                state[f"{prefix}op.{key}"] = value

        if self.left is not None:
            state.update(self.left.state_dict(prefix=f"{prefix}left."))

        if self.right is not None:
            state.update(self.right.state_dict(prefix=f"{prefix}right."))

        return state

    
    def visualize_tree_inorder(self, filename=None, format="png", leaf_transforms=None):
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
                a_val = node.operation.a.detach().cpu().flatten()[0].item()
                b_val = node.operation.b.detach().cpu().flatten()[0].item()
                label = f"{a_val:.3f} * {safe_op_name(node.operation.op)} + {b_val:.3f}"
            dot.node(node_id, label=label)

            if parent_id is not None:
                dot.edge(parent_id, node_id)

            if node.left:
                add_nodes_edges(node.left, node_id)
            if node.right:
                add_nodes_edges(node.right, node_id)

        add_nodes_edges(self)

        if filename is None:
            return dot

        dot.render(filename, format=format, cleanup=True)
        return dot
            
    
    def __str__(self, leaf_expressions=None):
        if self.operation_type == "leaf":
            if leaf_expressions is not None:
                return f"({leaf_expressions[self.leaf_idx]})"
            else:
                return f"x{self.leaf_idx}"
        elif self.operation_type == "unary":
            a_val = self.operation.a.detach().cpu().flatten()[0].item()
            b_val = self.operation.b.detach().cpu().flatten()[0].item()
            return f"({a_val:.3f} * {self.operation.op.__name__}({self.left.__str__(leaf_expressions)}) + {b_val:.3f})"
        elif self.operation_type == "binary":
            return f"({self.left.__str__(leaf_expressions)} {self.operation.op.__name__} {self.right.__str__(leaf_expressions)})"
