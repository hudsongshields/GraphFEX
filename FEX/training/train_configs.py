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
    num_groups: int = 1
    leaf_dim: int = None
    num_leaves: int = None

    lr: float = 0.001
    inter_lr: float = 0.001
    leaf_lr: float = None  # separate LR for leaf logits (defaults to lr if None)
    num_epochs: int = 15

    bfgs_epochs: int = 15
    bfgs_lr: float = 0.1

    tau_start: float = 1.0
    tau_end: float = 0.02

    # Multi-step rollout loss
    rollout_steps: int = 5        # Euler steps per rollout window
    rollout_weight: float = 0.8    # weight of rollout loss (ramped up over warmup)
    rollout_dt: float = 0.01       # timestep for Euler integration

    # Regularization
    mag_entropy_weight: float = 0.03  # weight for magnitude entropy regularization
    
    def __post_init__(self):
        self.tau_anneal_epochs = int(self.num_epochs * 0.75)
        self.set_hard_at = int(self.num_epochs * 0.5)
        self.leaf_lr = self.lr if self.leaf_lr is None else self.leaf_lr


runtimeconfig = RunTimeConfig()