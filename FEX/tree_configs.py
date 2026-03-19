from ast import List
from dataclasses import dataclass

from .nodes import Node
from .operations import UNARY_OPS, BINARY_OPS

# API only allows access to tree configs
__all__ = [
    "TREE_CONFIGS",
    "get_tree_config",
]

# Tree Configuration Returns Parent Node
def depth_2_tree(sample_indices) -> Node:
    leaf1 = Node(operation_type="leaf", leaf_idx=0, name="leaf1", operation=None)
    leaf2 = Node(operation_type="leaf", leaf_idx=1, name="leaf2", operation=None)
    branch1 = Node(operation_type="unary", operation=UNARY_OPS[sample_indices[1]], left=leaf1, name="branch1")
    branch2 = Node(operation_type="unary", operation=UNARY_OPS[sample_indices[2]], left=leaf2, name="branch2")
    parent_node = Node(operation_type="binary", operation=BINARY_OPS[sample_indices[0]], left=branch1, right=branch2, name="parent_node")
    return parent_node

def ops_per_depth_2_tree():
    return [len(BINARY_OPS), len(UNARY_OPS), len(UNARY_OPS)]

"""
Depth 3 Tree with:
- 1 binary op at the root
- 2 binary ops at the first level
- 4 unary ops at the second level
- 4 leaf nodes at the third level
"""
def depth_3_leaves_4(sample_indices) -> Node:
    
    leaf1 = Node(operation_type="leaf", leaf_idx=0, name="leaf1", operation=None)
    leaf2 = Node(operation_type="leaf", leaf_idx=1, name="leaf2", operation=None)
    leaf3 = Node(operation_type="leaf", leaf_idx=2, name="leaf3", operation=None)
    leaf4 = Node(operation_type="leaf", leaf_idx=3, name="leaf4", operation=None)

    l2_branch1 = Node(operation_type="unary", operation=UNARY_OPS[sample_indices[3]], left=leaf1, name="l2_branch1")
    l2_branch2 = Node(operation_type="unary", operation=UNARY_OPS[sample_indices[4]], left=leaf2, name="l2_branch2")
    l2_branch3 = Node(operation_type="unary", operation=UNARY_OPS[sample_indices[5]], left=leaf3, name="l2_branch3")
    l2_branch4 = Node(operation_type="unary", operation=UNARY_OPS[sample_indices[6]], left=leaf4, name="l2_branch4")

    l1_branch1 = Node(operation_type="binary", operation=BINARY_OPS[sample_indices[1]], left=l2_branch1, right=l2_branch2, name="l1_branch1")
    l1_branch2 = Node(operation_type="binary", operation=BINARY_OPS[sample_indices[2]], left=l2_branch3, right=l2_branch4, name="l1_branch2")

    parent_node = Node(operation_type="binary", operation=BINARY_OPS[sample_indices[0]], left=l1_branch1, right=l1_branch2, name="parent_node")

    return parent_node

def ops_per_depth_3_leaves_4():
    return [len(BINARY_OPS), len(BINARY_OPS), len(BINARY_OPS), len(UNARY_OPS), len(UNARY_OPS), len(UNARY_OPS), len(UNARY_OPS)]


"""
Depth 3 Tree:
- 1 unary op at the root
- 1 binary op at the first level
- 2 unary ops at the second level
- 2 leaf nodes
"""
def depth_3(sample_indices) -> Node:
    leaf1 = Node(operation_type="leaf", leaf_idx=0, name="leaf1", operation=None)
    leaf2 = Node(operation_type="leaf", leaf_idx=1, name="leaf2", operation=None)

    l2_branch1 = Node(operation_type="unary", operation=UNARY_OPS[sample_indices[2]], left=leaf1, name="l2_branch1")
    l2_branch2 = Node(operation_type="unary", operation=UNARY_OPS[sample_indices[3]], left=leaf2, name="l2_branch2")

    l1_branch1 = Node(operation_type="binary", operation=BINARY_OPS[sample_indices[1]], left=l2_branch1, right=l2_branch2, name="l1_branch1")

    parent_node = Node(operation_type="unary", operation=UNARY_OPS[sample_indices[0]], left=l1_branch1, name="parent_node")

    return parent_node

def ops_per_depth_3():
    return [len(UNARY_OPS), len(BINARY_OPS), len(UNARY_OPS), len(UNARY_OPS)]

"""
Depth 4 Tree:
- 1 binary op at the root
- 2 unary ops at the first level
- 2 binary ops at the second level
- 4 unary ops at the third level
- 4 leaf ops at the fourth level
"""
def depth_4(sample_indices) -> Node:
    leaf1 = Node(operation_type="leaf", leaf_idx=0, name="leaf1", operation=None)
    leaf2 = Node(operation_type="leaf", leaf_idx=1, name="leaf2", operation=None)
    leaf3 = Node(operation_type="leaf", leaf_idx=2, name="leaf3", operation=None)
    leaf4 = Node(operation_type="leaf", leaf_idx=3, name="leaf4", operation=None)

    l3_branch1 = Node(operation_type="unary", operation=UNARY_OPS[sample_indices[3]], left=leaf1, name="l3_branch1")
    l3_branch2 = Node(operation_type="unary", operation=UNARY_OPS[sample_indices[4]], left=leaf2, name="l3_branch2")
    l3_branch3 = Node(operation_type="unary", operation=UNARY_OPS[sample_indices[5]], left=leaf3, name="l3_branch3")
    l3_branch4 = Node(operation_type="unary", operation=UNARY_OPS[sample_indices[6]], left=leaf4, name="l3_branch4")

    l2_branch1 = Node(operation_type="binary", operation=BINARY_OPS[sample_indices[1]], left=l3_branch1, right=l3_branch2, name="l2_branch1")
    l2_branch2 = Node(operation_type="binary", operation=BINARY_OPS[sample_indices[2]], left=l3_branch3, right=l3_branch4, name="l2_branch2")

    l1_branch1 = Node(operation_type="unary", operation=UNARY_OPS[sample_indices[7]], left=l2_branch1, name="l1_branch1")
    l1_branch2 = Node(operation_type="unary", operation=UNARY_OPS[sample_indices[8]], left=l2_branch2, name="l1_branch2")

    parent_node = Node(operation_type="binary", operation=BINARY_OPS[sample_indices[0]], left=l1_branch1, right=l1_branch2, name="parent_node")

    return parent_node

def ops_per_depth_4():
    return [
        len(BINARY_OPS),  # parent_node
        len(BINARY_OPS),  # l2_branch1
        len(BINARY_OPS),  # l2_branch2
        len(UNARY_OPS),   # l3_branch1
        len(UNARY_OPS),   # l3_branch2
        len(UNARY_OPS),   # l3_branch3
        len(UNARY_OPS),   # l3_branch4
        len(UNARY_OPS),   # l1_branch1
        len(UNARY_OPS),   # l1_branch2
    ]

    

@dataclass 
class TreeConfig:
    tree_func: callable
    ops_per_node: List
    num_leaves:int

    def build_tree(self, sample_indices):
        return self.tree_func(sample_indices)


depth_2_tree_config = TreeConfig(tree_func=depth_2_tree, ops_per_node=ops_per_depth_2_tree(), num_leaves=2)
depth_3_tree_config = TreeConfig(tree_func=depth_3, ops_per_node=ops_per_depth_3(), num_leaves=2)
depth_4_tree_config = TreeConfig(tree_func=depth_4, ops_per_node=ops_per_depth_4(), num_leaves=4)
depth_3_leaves_4_config = TreeConfig(tree_func=depth_3_leaves_4, ops_per_node=ops_per_depth_3_leaves_4(), num_leaves=4)

TREE_CONFIGS = {
    "depth_2_tree_config": depth_2_tree_config,
    "depth_3_tree_config": depth_3_tree_config,
    "depth_4_tree_config": depth_4_tree_config,
    "depth_3_leaves_4_config": depth_3_leaves_4_config,
}


def get_tree_config(name: str):
    if name not in TREE_CONFIGS:
        raise KeyError(f"Unknown tree config name: {name}")
    return TREE_CONFIGS[name]
        