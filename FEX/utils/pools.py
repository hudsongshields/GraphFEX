from dataclasses import dataclass
from pathlib import Path
import re
import shutil
from typing import TYPE_CHECKING

import torch
from torch.multiprocessing import Lock

if TYPE_CHECKING:
    from ..models.nodes import Node


@dataclass
class PoolCandidate:
    node: "Node"
    reward: float
    id: int


@dataclass
class GraphPoolCandidate:
    inter_tree: object
    forcing_tree: object
    reward: float
    id: int
    metadata: dict = None


class Pool:
    def __init__(self, pool_size: int):
        self.pool = []
        self.pool_size = pool_size
        self.threshold = 0.0

    def add_new(self, candidate: PoolCandidate):
        reward = candidate.reward
        should_add = len(self.pool) < self.pool_size or reward > self.threshold
        if not should_add:
            return

        self.pool.append(candidate)
        self.sort()
        if len(self.pool) > self.pool_size:
            self.pool.pop(-1)
        self.update_threshold()

    def update_threshold(self):
        self.threshold = self.pool[-1].reward if len(self.pool) == self.pool_size else 0.0

    def sort(self):
        self.pool.sort(key=lambda x: x.reward, reverse=True)

    def __iter__(self):
        return iter(self.pool)

    def __len__(self):
        return len(self.pool)


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

        op1 = getattr(getattr(node1, "operation", None), "op", None)
        op2 = getattr(getattr(node2, "operation", None), "op", None)
        if getattr(op1, "__name__", None) != getattr(op2, "__name__", None):
            return False

        if (node1.left is None) != (node2.left is None):
            return False
        if (node1.right is None) != (node2.right is None):
            return False

        if node1.left is not None and not self.check_node_equality(node1.left, node2.left):
            return False
        if node1.right is not None and not self.check_node_equality(node1.right, node2.right):
            return False

        return True

    def is_unique(self, candidate: GraphPoolCandidate):
        for idx, existing in enumerate(self.pool):
            same_inter_tree = self.check_node_equality(
                existing.inter_tree.parent_node,
                candidate.inter_tree.parent_node,
            )
            same_forcing_tree = self.check_node_equality(
                existing.forcing_tree.parent_node,
                candidate.forcing_tree.parent_node,
            )

            if same_inter_tree and same_forcing_tree:
                if candidate.reward > existing.reward:
                    self.pool[idx] = candidate
                    self.sort()
                    self.update_threshold()
                return False

        return True

    def update_threshold(self):
        self.threshold = self.pool[-1].reward if len(self.pool) == self.pool_size else 0.0

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

    def __len__(self):
        return len(self.pool)

    @staticmethod
    def _extract_state(ckpt):
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            return ckpt["state_dict"]
        return ckpt

    @staticmethod
    def _meta_value(ckpt, state, key, default=None):
        if isinstance(ckpt, dict) and key in ckpt:
            return ckpt[key]
        if isinstance(state, dict) and key in state:
            return state[key]
        return default

    @staticmethod
    def _infer_leaf_dim(ckpt, state):
        leaf_dim = GraphPool._meta_value(ckpt, state, "_meta_leaf_dim")
        if leaf_dim is not None:
            return int(leaf_dim)

        for key, value in state.items():
            if re.fullmatch(r"leaf_mlps\.\d+\.logits", key):
                return value.numel()

        raise ValueError("Could not infer leaf_dim from checkpoint.")

    @staticmethod
    def _infer_num_leaves(ckpt, state):
        num_leaves = GraphPool._meta_value(ckpt, state, "_meta_num_leaves")
        if num_leaves is not None:
            return int(num_leaves)

        leaf_indices = []
        for key in state.keys():
            match = re.match(r"leaf_mlps\.(\d+)\.", key)
            if match:
                leaf_indices.append(int(match.group(1)))

        if not leaf_indices:
            raise ValueError("Could not infer num_leaves from checkpoint.")

        return max(leaf_indices) + 1

    @staticmethod
    def _sample_indices(ckpt, state):
        sample_indices = GraphPool._meta_value(ckpt, state, "_meta_sample_indices")
        if sample_indices is None:
            return None
        return torch.tensor([int(i) for i in sample_indices], dtype=torch.long)

    @classmethod
    def load_candidates(cls, directory, forcing_tree_config=None, inter_tree_config=None, device=None):
        directory = Path(directory)
        inter_files = sorted(directory.glob("inter_tree*.pt"))
        candidates = []

        from FEX.models.learnable_tree import FEX

        for inter_file in inter_files:
            match = re.search(r"inter_tree(\d+)\.pt$", inter_file.name)
            if not match:
                continue

            cand_id = int(match.group(1))
            forcing_file = directory / f"forcing_tree{cand_id}.pt"
            if not forcing_file.exists():
                continue

            inter_ckpt = torch.load(inter_file, map_location=device or "cpu")
            forcing_ckpt = torch.load(forcing_file, map_location=device or "cpu")
            inter_state = cls._extract_state(inter_ckpt)
            forcing_state = cls._extract_state(forcing_ckpt)

            inter_tree = FEX(
                leaf_dim=cls._infer_leaf_dim(inter_ckpt, inter_state),
                num_leaves=cls._infer_num_leaves(inter_ckpt, inter_state),
                tree_structure=inter_tree_config,
                sample_indices=cls._sample_indices(inter_ckpt, inter_state),
            )
            forcing_tree = FEX(
                leaf_dim=cls._infer_leaf_dim(forcing_ckpt, forcing_state),
                num_leaves=cls._infer_num_leaves(forcing_ckpt, forcing_state),
                tree_structure=forcing_tree_config,
                sample_indices=cls._sample_indices(forcing_ckpt, forcing_state),
            )

            inter_tree.load_state_dict(inter_state, strict=False)
            forcing_tree.load_state_dict(forcing_state, strict=False)

            if device is not None:
                inter_tree.to(device)
                forcing_tree.to(device)

            candidates.append(
                GraphPoolCandidate(
                    inter_tree=inter_tree,
                    forcing_tree=forcing_tree,
                    reward=0.0,
                    id=cand_id,
                    metadata={"inter_tree": inter_ckpt, "forcing_tree": forcing_ckpt},
                )
            )

        pool = cls(pool_size=max(1, len(candidates)))
        for candidate in candidates:
            pool.add_new(candidate)
        return pool

    def _build_fex_checkpoint(self, fex):
        return {"state_dict": fex.state_dict()}

    def save_candidates(self, directory: str, clear_directory: bool = True):
        target = Path(directory)
        if clear_directory and target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)

        for candidate in self.pool:
            torch.save(self._build_fex_checkpoint(candidate.inter_tree), target / f"inter_tree{candidate.id}.pt")
            torch.save(self._build_fex_checkpoint(candidate.forcing_tree), target / f"forcing_tree{candidate.id}.pt")

    def visualize_candidates(self, directory: str = "candidates_viz", clear_directory: bool = True):
        target = Path(directory)
        if clear_directory and target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)

        for candidate in self.pool:
            candidate.inter_tree.visualize_tree(directory=str(target / f"inter_tree{candidate.id}"))
            candidate.forcing_tree.visualize_tree(directory=str(target / f"forcing_tree{candidate.id}"))
