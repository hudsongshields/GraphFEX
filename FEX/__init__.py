"""API for the FEX package"""

from .models.controllers import Controller
from .models.learnable_tree import FEX
from .training.train_configs import ControllerConfig, FEXConfig, RunTimeConfig, runtimeconfig
from . import utils
from .utils.tree_configs import get_tree_config


def train_network_controller(*args, **kwargs):
	from .training.train_controller import train_network_controller as _train_network_controller
	return _train_network_controller(*args, **kwargs)


def train_network_fex(*args, **kwargs):
	from .training.train_fex import train_network_fex as _train_network_fex
	return _train_network_fex(*args, **kwargs)

__all__ = [
	"Controller",
	"ControllerConfig",
	"FEX",
	"FEXConfig",
	"RunTimeConfig",
	"runtimeconfig",
	"train_network_controller",
	"train_network_fex",
    "get_tree_config",
]

