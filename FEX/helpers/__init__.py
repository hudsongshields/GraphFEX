from . import sampler
from . import pools
from . import tree_configs
from . import operations

__all__ = [
    "sampler",
    "numerical_deriv",
    "pools",
    "tree_configs",
    "operations",
]


def __getattr__(name):
    if name == "numerical_deriv":
        from importlib import import_module

        module = import_module(f"{__name__}.numerical_deriv")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
