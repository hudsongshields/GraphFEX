from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models.nodes import Node
from dataclasses import dataclass
from pathlib import Path
import shutil
import torch
import re

@dataclass
class PoolCandidate:
    tree: "Node"
    reward: float
    id: int


class Pool():
    def __init__(self, pool_size: int):
        self.pool = []
        self.pool_size = pool_size
        self.threshold = 0.0
    def add_new(self, candidate: PoolCandidate):
        _, reward = candidate.tree, candidate.reward
        should_add = len(self.pool) < self.pool_size or reward > self.threshold
        if not should_add:
            return
        self.pool.append(candidate)
        self.sort()
        if len(self.pool) > self.pool_size:
            self.pool.pop(-1)
        if len(self.pool) == self.pool_size:
            self.threshold = self.pool[-1].reward
        else:
            self.threshold = 0.0
    def sort(self):
        self.pool.sort(key=lambda x: x.reward, reverse=True)
    def __iter__(self):
        return iter(self.pool)

    

from threading import Lock
@dataclass
class GraphPoolCandidate:
    inter_tree: object
    forcing_tree: object
    reward: float
    id: int
    metadata: dict = None

class GraphPool:
    def __init__(self, pool_size: int):
        self.pool = []
        self.pool_size = pool_size
        self.threshold = 0.0
        self.lock = Lock()

    def check_node_equality(self, node1, node2):
        if node1.operation_type != node2.operation_type:
            return False
        
        if node1.operation_type == "leaf" and node2.operation_type == "leaf":
            return True

        if node1.op.__name__ != node2.op.__name__:
            return False

        if (node1.left is None) != (node2.left is None):
            return False

        if (node1.right is None) != (node2.right is None):
            return False

        if node1.left is not None:
            if not self.check_node_equality(node1.left, node2.left):
                return False

        if node1.right is not None:
            if not self.check_node_equality(node1.right, node2.right):
                return False

        return True


    def is_unique(self, candidate: GraphPoolCandidate):

        for i, existing in enumerate(self.pool):
            same_inter_tree = self.check_node_equality(existing.inter_tree.parent_node, candidate.inter_tree.parent_node)
            same_forcing_tree = self.check_node_equality(existing.forcing_tree.parent_node, candidate.forcing_tree.parent_node)

            if same_inter_tree and same_forcing_tree:
                if candidate.reward > existing.reward:
                    self.pool[i] = candidate
                    self.sort()
                    self.update_threshold()

                return False

        return True


    def update_threshold(self):
        if len(self.pool) == self.pool_size:
            self.threshold = self.pool[-1].reward
        else:
            self.threshold = 0.0


    def add_new(self, candidate: GraphPoolCandidate, check_if_unique: bool = False):
        with self.lock:
            reward = candidate.reward
            should_add = len(self.pool) < self.pool_size or reward > self.threshold

            if not should_add:
                return

            if check_if_unique and not self.is_unique(candidate):
                return

            self.pool.append(candidate)
            self.sort()

            if len(self.pool) > self.pool_size:
                self.pool.pop(-1)

            self.update_threshold()
                
    def sort(self):
        self.pool.sort(key=lambda x: x.reward, reverse=True)
    def __iter__(self):
        return iter(self.pool)
    