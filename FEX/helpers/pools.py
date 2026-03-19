from ..nodes import Node
from dataclasses import dataclass
from pathlib import Path
import shutil
import torch

@dataclass
class PoolCandidate:
    node: Node
    reward: float
    id: int

@dataclass
class GraphPoolCandidate:
    inter_tree: Node
    forcing_tree: Node
    reward: float
    id: int
    metadata: dict = None


class Pool():
    def __init__(self, pool_size: int):
        self.pool = []
        self.pool_size = pool_size

        self.threshold = 0.0
        
    def add_new(self, candidate: PoolCandidate):
        _, reward = candidate.node, candidate.reward
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
    
class GraphPool(Pool):
    def _build_fex_checkpoint(self, fex):
        sample_indices = fex.sample_indices
        if isinstance(sample_indices, torch.Tensor):
            sample_indices = sample_indices.detach().cpu().to(dtype=torch.int64).tolist()
        elif sample_indices is not None:
            sample_indices = [int(i) for i in sample_indices]

        return {
            "state_dict": fex.state_dict(),
            "sample_indices": sample_indices,
            "tree_config_name": fex._tree_config_name(),
            "leaf_dim": int(fex.leaf_dim),
            "num_leaves": int(len(fex.leaf_mlps)),
        }

    def add_new(self, candidate: GraphPoolCandidate):
        reward = candidate.reward
        should_add = len(self.pool) < self.pool_size or reward > self.threshold
        if not should_add:
            return

        candidate.metadata = {
            "inter_tree": self._build_fex_checkpoint(candidate.inter_tree),
            "forcing_tree": self._build_fex_checkpoint(candidate.forcing_tree),
        }
        self.pool.append(candidate)
        self.sort()

        if len(self.pool) > self.pool_size:
            self.pool.pop(-1)

        if len(self.pool) == self.pool_size:
            self.threshold = self.pool[-1].reward

    # Tools to visualize and store candidates in the pool.
    def _prepare_directory(self, directory: str, clear: bool = False) -> Path:
        target = Path(directory)
        if clear and target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)
        return target

    def save_candidates(self, directory: str, clear_directory: bool = True):
        target = self._prepare_directory(directory, clear=clear_directory)
        for cand in self.pool:
            torch.save(cand.metadata["inter_tree"], target / f"inter_tree{cand.id}.pt")
            torch.save(cand.metadata["forcing_tree"], target / f"forcing_tree{cand.id}.pt")

    def visualize_candidates(self, directory: str = "candidates_viz", clear_directory: bool = True):
        target = self._prepare_directory(directory, clear=clear_directory)
        for cand in self.pool:
            cand.inter_tree.visualize_tree(filename=str(target / f"inter_tree{cand.id}"))
            cand.forcing_tree.visualize_tree(filename=str(target / f"forcing_tree{cand.id}"))


if __name__ == "__main__":
    dummy_operation = lambda x: x
    pool = Pool(pool_size=3)
    pool.add_new(PoolCandidate(node=Node(operation=dummy_operation, operation_type='unary'), reward=0.5))
    pool.add_new(PoolCandidate(node=Node(operation=dummy_operation, operation_type='unary'), reward=0.7))
    pool.add_new(PoolCandidate(node=Node(operation=dummy_operation, operation_type='unary'), reward=0.6))
    pool.add_new(PoolCandidate(node=Node(operation=dummy_operation, operation_type='unary'), reward=0.4))

    for candidate in pool:
        print(candidate.reward)