from dataclasses import dataclass
import torch

@dataclass
class RunTimeConfig:
    device: str="cpu"

    def __post_init__(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        print(f"Using device: {self.device}")


@dataclass
class ControllerConfig():
    input_dim: int = 10
    hidden_dim: int = 20
    lr: float = 0.001
    num_epochs: int = 100
    num_trees: int = 2 # for forcing and interaction trees

    num_cands_per_epoch: int = 10
    percentile_threshold: float = 0.5

@dataclass
class FEXConfig():
    lr: float = 0.001
    num_epochs: int = 10
    max_nodes: int = 20
    leaf_dim: int = 1
    num_leaves: int = 2
    bfgs_lr: float = 1e-4
    max_norm: float = 1.0
    tau_start: float = 1.0
    tau_end: float = 0.1
    tau_anneal_epochs: int = 10


runtimeconfig = RunTimeConfig()