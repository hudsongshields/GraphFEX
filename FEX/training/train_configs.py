from dataclasses import dataclass
import torch

from pathlib import Path
import logging


@dataclass
class ControllerConfig():
    input_dim: int = 20
    hidden_dim: int = 64
    lr: float = 0.01
    num_epochs: int = 100

    num_cands_per_epoch: int = 10
    percentile_threshold: float = 0.5
    poolsize: int = 10

    epsilon_greedy: float = 0.3

@dataclass
class FEXConfig():
    target_dim: int = 0
    expression_threshold: float = 1e-3


    lr: float = 0.02
    inter_lr: float = 0.008
    num_epochs: int = 30
    lr_decay: bool = False

    bfgs_epochs: int = 15
    bfgs_lr: float = 0.8


