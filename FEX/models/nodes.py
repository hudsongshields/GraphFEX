import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Callable, Optional
import copy



class UnaryOperation(nn.Module):
    def __init__(self, op: Callable):
        super().__init__()
        self.scale = 1.5

        self.a = nn.Parameter(torch.randn((1)) * self.scale)
        self.b = nn.Parameter(torch.randn((1)) * self.scale)
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
    
    def reset(self):
        if isinstance(self.operation, nn.Module):
            if self.operation_type == "unary":
                if hasattr(self.operation, 'a'):
                    self.operation.a.data.copy_(torch.randn_like(self.operation.a) * getattr(self.operation, 'scale', 1.0))
                if hasattr(self.operation, 'b'):
                    self.operation.b.data.copy_(torch.randn_like(self.operation.b) * getattr(self.operation, 'scale', 1.0))
        if self.left is not None:
            self.left.reset()
        if self.right is not None:
            self.right.reset()
    
    def copy_inorder(self):
        # Avoid deepcopy on nn.Modules and tensors; use state_dict/load_state_dict for safe copying
        device = None
        if isinstance(self.operation, nn.Module):
            # Get device from first parameter
            params = list(self.operation.parameters())
            if params:
                device = params[0].device
        if self.operation_type == "unary" and isinstance(self.operation, UnaryOperation):
            new_op = UnaryOperation(self.operation.op)
            new_op.load_state_dict(self.operation.state_dict())
            if device is not None:
                new_op.to(device)
        elif self.operation_type == "binary" and isinstance(self.operation, BinaryOperation):
            new_op = BinaryOperation(self.operation.op)
            new_op.load_state_dict(self.operation.state_dict())
            if device is not None:
                new_op.to(device)
        else:
            new_op = self.operation # for leaves or non-module ops

        node = Node(
            operation_type=self.operation_type,
            operation=new_op,
            leaf_idx=self.leaf_idx,
            left=self.left.copy_inorder() if self.left else None,
            right=self.right.copy_inorder() if self.right else None,
            name=self.name
        )
        # Propagate device to children
        if device is not None:
            node.to(device)
        return node

        
    def __str__(self, leaf_expressions=None):
        if self.operation_type == "leaf":
            if leaf_expressions is not None:
                return f"({leaf_expressions[self.leaf_idx]})"
            else:
                return f"x{self.leaf_idx}"
        elif self.operation_type == "unary":
            a, b = self.operation.a.detach().item(), self.operation.b.detach().item()
            return f"({a:.3f} * {self.operation.op.__name__}({self.left.__str__(leaf_expressions)}) + {b:.3f})"
        elif self.operation_type == "binary":
            return f"({self.left.__str__(leaf_expressions)} {self.operation.op.__name__} {self.right.__str__(leaf_expressions)})"
        
