import torch
from typing import Callable


EPS = 1e-6
POLY_CLAMP = 5.0


def identity(x):
    return x


def square(x):
    x = torch.clamp(x, min=-POLY_CLAMP, max=POLY_CLAMP)
    return torch.square(x)


def cube(x):
    x = torch.clamp(x, min=-POLY_CLAMP, max=POLY_CLAMP)
    return torch.pow(x, 3)


def fourth_power(x):
    x = torch.clamp(x, min=-POLY_CLAMP, max=POLY_CLAMP)
    return torch.pow(x, 4)


def safe_exp(x):
    # Keep exponential growth bounded enough for stable optimization.
    return torch.exp(torch.clamp(x, min=-10.0, max=10.0))


def sin(x):
    return torch.sin(x)


def safe_reciprocal(x):
    denom = torch.where(torch.abs(x) < EPS, torch.full_like(x, EPS), x)
    return torch.reciprocal(denom)


def add(x, y):
    return torch.add(x, y)


def mul(x, y):
    return torch.mul(x, y)


def sub(x, y):
    return torch.sub(x, y)


def safe_div(x, y):
    denom = torch.where(torch.abs(y) < EPS, torch.full_like(y, EPS), y)
    return torch.div(x, denom)

UNARY_OPS = [
    identity,
    square,
    cube,
    fourth_power,
    safe_exp,
    sin,
    safe_reciprocal,
]

BINARY_OPS = [
    add,
    mul,
    sub,
    safe_div,
]