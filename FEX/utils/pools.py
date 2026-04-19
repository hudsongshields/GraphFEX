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
    node: "Node"
    reward: float
    id: int

@dataclass
class GraphPoolCandidate:
    inter_tree: "Node"
    forcing_tree: "Node"
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

# --- GraphPool and GraphPoolCandidate with robust load_candidates ---
from dataclasses import dataclass
from pathlib import Path
import shutil
import torch
import re
from torch.multiprocessing import Lock

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

    def add_new(self, candidate: GraphPoolCandidate):
        reward = candidate.reward
        should_add = len(self.pool) < self.pool_size or reward > self.threshold
        if not should_add:
            return
        
        with self.lock:
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
    
    @classmethod
    def load_candidates(cls, directory, forcing_tree_config, inter_tree_config, device=None):
        directory = Path(directory)
        inter_files = list(directory.glob("inter_tree*.pt"))
        candidates = []

        def extract_state(ckpt):
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                return ckpt["state_dict"]
            return ckpt

        def infer_leaf_dim(ckpt, state):
            if isinstance(ckpt, dict) and "_meta_leaf_dim" in ckpt:
                return ckpt["_meta_leaf_dim"]

            # logits length = feature/input dimension for each leaf MLP
            for k, v in state.items():
                if re.fullmatch(r"leaf_mlps\.\d+\.logits", k):
                    return v.numel()

            raise ValueError("Could not infer leaf_dim from checkpoint.")

        def infer_num_leaves(ckpt, state):
            if isinstance(ckpt, dict) and "_meta_num_leaves" in ckpt:
                return ckpt["_meta_num_leaves"]

            idxs = []
            for k in state.keys():
                m = re.match(r"leaf_mlps\.(\d+)\.", k)
                if m:
                    idxs.append(int(m.group(1)))

            if not idxs:
                raise ValueError("Could not infer num_leaves from checkpoint.")

            return max(idxs) + 1

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

            from FEX.models.learnable_tree import FEX

            inter_state = extract_state(inter_ckpt)
            forcing_state = extract_state(forcing_ckpt)

            inter_leaf_dim = infer_leaf_dim(inter_ckpt, inter_state)
            forcing_leaf_dim = infer_leaf_dim(forcing_ckpt, forcing_state)

            inter_num_leaves = infer_num_leaves(inter_ckpt, inter_state)
            forcing_num_leaves = infer_num_leaves(forcing_ckpt, forcing_state)

            inter_tree = FEX(
                leaf_dim=inter_leaf_dim,
                num_leaves=inter_num_leaves,
                tree_structure=inter_tree_config.tree_func,
                sample_indices=inter_ckpt.get("_meta_sample_indices", None) if isinstance(inter_ckpt, dict) else None,
            )

            forcing_tree = FEX(
                leaf_dim=forcing_leaf_dim,
                num_leaves=forcing_num_leaves,
                tree_structure=forcing_tree_config.tree_func,
                sample_indices=forcing_ckpt.get("_meta_sample_indices", None) if isinstance(forcing_ckpt, dict) else None,
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

        pool = cls(pool_size=len(candidates))
        for cand in candidates:
            pool.add_new(cand)
        return pool
    
    def _build_fex_checkpoint(self, fex):
        return {"state_dict": fex.state_dict()}
    def save_candidates(self, directory: str, clear_directory: bool = True):
        target = Path(directory)
        if clear_directory and target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)
        for cand in self.pool:
            torch.save(self._build_fex_checkpoint(cand.inter_tree), target / f"inter_tree{cand.id}.pt")
            torch.save(self._build_fex_checkpoint(cand.forcing_tree), target / f"forcing_tree{cand.id}.pt")
    def visualize_candidates(self, directory: str = "candidates_viz", clear_directory: bool = True):
        target = Path(directory)
        if clear_directory and target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)
        for cand in self.pool:
            cand.inter_tree.visualize_tree(directory=str(target / f"inter_tree{cand.id}"))
            cand.forcing_tree.visualize_tree(directory=str(target / f"forcing_tree{cand.id}"))
