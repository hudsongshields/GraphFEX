from dataclasses import dataclass
import torch

from pathlib import Path
import logging

@dataclass
class RunTimeConfig:
    device: str="cpu"
    train_logger: logging.Logger = None

    def __post_init__(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        print(f"Using device: {self.device}")

    def CreateLogger(self, log_path: str, name: str="train_logger"):
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        fh = logging.FileHandler(log_path, mode="w")
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

        self.train_logger = logger



@dataclass
class ControllerConfig():
    input_dim: int = 10
    hidden_dim: int = 20
    lr: float = 0.01
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
    bfgs_lr: float = 0.1
    
    tau_start: float = 1.0
    tau_end: float = 0.1
    tau_anneal_epochs: int = 10


runtimeconfig = RunTimeConfig()