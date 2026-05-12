from dataclasses import dataclass
import torch

from pathlib import Path
import logging

@dataclass
class RunTimeConfig:
    device: str="cpu"
    train_logger: logging.Logger = None
    train_log_path: str = None

    def __post_init__(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        print(f"Using device: {self.device}")

    def CreateLogger(self, log_path: str, name: str="train_logger", mode: str = "w"):
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        if logger.handlers:
            for handler in list(logger.handlers):
                logger.removeHandler(handler)
                handler.close()

        fh = logging.FileHandler(log_path, mode=mode)
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

        self.train_logger = logger
        self.train_log_path = str(log_path)



@dataclass
class ControllerConfig():
    input_dim: int = 10
    hidden_dim: int = 20
    lr: float = 0.01
    num_epochs: int = 100
    num_trees: int = 2 # for forcing and interaction trees

    num_cands_per_epoch: int = 10
    percentile_threshold: float = 0.5
    poolsize: int = 10

    epsilon_greedy: float = 0.1

@dataclass
class FEXConfig():
    num_groups: int = 1
    leaf_dim: int = None
    num_leaves: int = None
    leaf_entropy_weight: float = 0.8
    decay_entropy_until: float = 1.0
    set_hard_at: float = 1.1

    lr: float = 0.02
    inter_lr: float = 0.008
    num_epochs: int = 30
    lr_decay: float = 0.0

    bfgs_epochs: int = 15
    bfgs_lr: float = 0.8

    tau_start: float = 8.0
    tau_end: float = 4.0

    # Regularization
    mag_entropy_weight: float = 0.0 # weight for magnitude entropy regularization
    pct_cosine_restart: float = 1.0
    
    def __post_init__(self):
        self.tau_anneal_epochs = self.num_epochs
        self.set_hard_at_epoch = int(self.num_epochs * self.set_hard_at) # default: Dont set hard


runtimeconfig = RunTimeConfig()