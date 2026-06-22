
__all__ = ["FEX"]

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
        self.logits = nn.Parameter(torch.randn(input_dim) * 0.1)
        self.register_buffer("logit_mask", torch.zeros(input_dim))
        
    
    def forward(self, leaf_input: torch.Tensor):
        # generate random val from uniform dis
        prob = torch.rand(1).item()
        logit_push = torch.zeros_like(self.logits)
        if self.training and prob > 0.9:
            random_idx = torch.randint(0, self.logits.size(0), (1,))
            logit_push[random_idx] = 10.0
        probs = torch.softmax(self.logits + self.logit_mask + logit_push, dim=-1)
        return torch.sum(leaf_input * probs, dim=-1, keepdim=True)

    def reset_parameters(self):
        with torch.no_grad():
            self.logits.normal_(mean=0.0, std=0.1)

    """ Gumbel softmax control """
    def set_hard(self, hard: bool):
        pass

    def set_tau(self, tau: float):
        pass

    def get_tau(self):
        return 0.0

    
    """ Printable expression """
    def _selected_dim(self):
        return int((self.logits + self.logit_mask).detach().argmax().item())

    def _selection_confidence(self):
        # Confidence proxy based on normalized absolute linear weights.
        masked = (self.logits + self.logit_mask).detach().abs()
        probs = torch.softmax(masked, dim=-1)
        return float(probs.max().item())

    def __str__(self):
        probs = torch.softmax((self.logits + self.logit_mask).detach(), dim=-1)
        selected_dim = int(probs.argmax().item())
        confidence = float(probs[selected_dim].item())
        return f"x[{selected_dim}]@{confidence:.3f}"
    

class FEX(nn.Module):
    def __init__(self, leaf_dim, num_leaves, sample_indices=None, tree_structure=None, parent_node=None, init_tau=5.0, hard=False, **kwargs): 
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

        leaf_mlps = [LeafMLP(self.leaf_dim, init_tau=init_tau, hard=hard) for _ in range(num_leaves)]
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

    def reset(self, fex_config=None):
        if fex_config is not None and hasattr(fex_config, 'tau_start'):
            tau_value = float(fex_config.tau_start)
        else:
            tau_value = 5.0

        for leaf in self.leaf_mlps:
            if hasattr(leaf, 'reset_parameters'):
                leaf.reset_parameters()
            if hasattr(leaf, 'logits'):
                nn.init.normal_(leaf.logits, mean=0.0, std=0.1)
            if hasattr(leaf, 'tau'):
                leaf.tau.fill_(tau_value)
            if hasattr(leaf, 'hard'):
                leaf.hard = False

        # Reset all tree node parameters as in their constructor, but keep structure
        def action(node: Node):
            node.reset()
        traverse(self.parent_node, action)

    def leaf_params(self):
        """Parameters controlling leaf dimension selection (logits + sigma)."""
        for leaf_mlp in self.leaf_mlps:
            yield from leaf_mlp.parameters()

    def tree_params(self):
        """Parameters controlling tree node scalars (sign, magnitude, bias)."""
        return list(self.parent_node.get_parameters())
    
    def tree_mags(self):
        """Helper to extract 'a' parameters from tree nodes for regularization."""
        mags = []
        def action(node: Node):
            if node.operation_type == "unary":
                mags.append(node.operation.a)
        traverse(self.parent_node, action)
        return mags

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
        leaf_expressions = [str(leaf_mlp) for leaf_mlp in self.leaf_mlps]
        return self.parent_node.__str__(leaf_expressions=leaf_expressions)

    def symbolic_expression(self, variable_names=None, digits: int = 4, threshold: float = 1e-6):
        """Build and simplify the hard-selected FEX expression with SymPy."""
        import sympy as sp

        if variable_names is None:
            variable_names = [f"x{i + 1}" for i in range(self.leaf_dim)]
        if len(variable_names) != self.leaf_dim:
            raise ValueError(
                f"Expected {self.leaf_dim} variable names, got {len(variable_names)}."
            )

        symbols = sp.symbols(" ".join(variable_names))
        if self.leaf_dim == 1:
            symbols = (symbols,)

        leaf_expressions = [
            symbols[leaf._selected_dim()]
            for leaf in self.leaf_mlps
        ]

        def rounded_parameter(value):
            number = round(float(value.detach().item()), digits)
            return sp.Float(0.0 if abs(number) < threshold else number)

        def build(node):
            if node.operation_type == "leaf":
                return leaf_expressions[node.leaf_idx]

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

            child = build(node.left)
            op_name = node.operation.op.__name__
            if op_name == "identity":
                transformed = child
            elif op_name == "square":
                transformed = child ** 2
            elif op_name == "cube":
                transformed = child ** 3
            elif op_name == "fourth_power":
                transformed = child ** 4
            elif op_name == "safe_exp":
                transformed = sp.exp(child)
            elif op_name == "sigmoid":
                transformed = 1 / (1 + sp.exp(-child))
            elif op_name == "safe_reciprocal":
                transformed = 1 / child
            else:
                raise ValueError(f"Unsupported unary operator: {op_name}")

            return rounded_parameter(node.operation.a) * transformed + rounded_parameter(node.operation.b)

        expanded = sp.expand(build(self.parent_node))
        retained_terms = []
        for term in sp.Add.make_args(expanded):
            coefficient, _ = term.as_coeff_Mul()
            if not coefficient.is_number or abs(float(coefficient)) >= threshold:
                retained_terms.append(term)

        return sp.simplify(sp.Add(*retained_terms))

    def simplified_expression(self, variable_names=None, digits: int = 4, threshold: float = 1e-6):
        return str(self.symbolic_expression(variable_names, digits, threshold))
    
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
