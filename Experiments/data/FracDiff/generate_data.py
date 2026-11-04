"""1D space-fractional diffusion example from fPINNs (Pang, Lu & Karniadakis, 2019, Sec. 4.1.2).

    u_t = -c (-Delta)^{alpha/2} u + f(x, t),   x in (-1, 1),   u = 0 outside (-1, 1)

Exact solution and source (advection v = 0):

    u(x, t) = x (1 - x^2)^{1 + alpha/2} e^{-t}
    (-Delta)^{alpha/2}[x (1 - x^2)^{1 + alpha/2}] = Gamma(alpha + 3) / 6 * [3 - (3 + alpha) x^2] x
    f(x, t) = -u + c * Gamma(alpha + 3) / 6 * [3 - (3 + alpha) x^2] x e^{-t}

The source f is treated as known, as in the paper's inverse problem, so the
FEX target is u_t - f. The fractional Laplacian becomes graph coupling:

    -c (-Delta)^{alpha/2} u_i  ~  sum_{j != i} A_ij * c * (u_j - u_i)

with A_ij = -g_|i-j| / h^alpha over interior nodes, plus one ghost node on
each side whose state is always 0 and whose edge weights carry the pull of
the zero exterior. FEX should find the interaction c * (u_j - u_i) and a zero
forcing term.
"""

import math

import numpy as np
import torch


def fractional_weights(alpha: float, num: int) -> np.ndarray:
    """Fractional centred-difference weights g_0..g_num (g_k < 0 for k >= 1)."""
    g = np.empty(num + 1)
    g[0] = math.gamma(alpha + 1) / math.gamma(alpha / 2 + 1) ** 2
    for k in range(num):
        g[k + 1] = g[k] * (k - alpha / 2) / (alpha / 2 + k + 1)
    return g


def build_graph(num_intervals: int = 100, alpha: float = 1.5):
    """Grid on (-1, 1) as a weighted graph.

    Returns (x, adjacency, score_nodes):
      x            interior grid points, shape (num_intervals - 1,)
      adjacency    (n_int + 2, n_int + 2); the last two nodes are the left and
                   right ghosts (state always 0, no incoming edges)
      score_nodes  indices of the interior nodes
    """
    n = num_intervals
    h = 2.0 / n
    n_int = n - 1
    x = -1.0 + h * np.arange(1, n)
    g = fractional_weights(alpha, n)
    abs_g = np.abs(g)
    scale = h ** (-alpha)

    A = np.zeros((n_int + 2, n_int + 2))
    offsets = np.abs(np.arange(n_int)[:, None] - np.arange(n_int)[None, :])
    A[:n_int, :n_int] = -g[offsets] * scale
    np.fill_diagonal(A[:n_int, :n_int], 0.0)

    # Exterior tail: sum_{k >= m} |g_k| = g_0 / 2 - sum_{k=1}^{m-1} |g_k|  (exact, no truncation)
    head = np.concatenate([[0.0], np.cumsum(abs_g[1:])])  # head[m] = sum_{k=1}^{m} |g_k|
    tail = lambda m: g[0] / 2 - head[m - 1]
    left, right = n_int, n_int + 1
    for idx in range(n_int):
        i = idx + 1  # grid index of this interior node
        A[idx, left] = tail(i) * scale
        A[idx, right] = tail(n - i) * scale

    adjacency = torch.tensor(A, dtype=torch.float32)
    score_nodes = torch.arange(n_int, dtype=torch.long)
    return torch.tensor(x, dtype=torch.float32), adjacency, score_nodes


def exact_solution(x, t, alpha: float = 1.5):
    x = torch.as_tensor(x)[None, :]
    t = torch.as_tensor(t)[:, None]
    return x * (1 - x**2) ** (1 + alpha / 2) * torch.exp(-t)


def fractional_laplacian_of_profile(x, alpha: float = 1.5):
    """Closed form of (-Delta)^{alpha/2}[x (1 - x^2)^{1 + alpha/2}] on (-1, 1)."""
    return math.gamma(alpha + 3) / 6 * (3 - (3 + alpha) * x**2) * x


def source(x, t, alpha: float = 1.5, c: float = 1.0):
    x = torch.as_tensor(x)[None, :]
    t = torch.as_tensor(t)[:, None]
    profile = x * (1 - x**2) ** (1 + alpha / 2)
    return (-profile + c * fractional_laplacian_of_profile(x, alpha)) * torch.exp(-t)


def add_noise_db(u: torch.Tensor, snr_db: float, generator=None) -> torch.Tensor:
    """White Gaussian noise at snr_db, relative to the variance of u over all
    times and interior points (the spatial profile is the signal here)."""
    noise_std = torch.sqrt(u.var() / 10.0 ** (snr_db / 10.0))
    noise = torch.randn(u.shape, generator=generator, dtype=u.dtype)
    return u + noise_std * noise


def five_point(u: torch.Tensor, dt: float):
    du = (u[:-4] - 8 * u[1:-3] + 8 * u[3:-1] - u[4:]) / (12 * dt)
    return u[2:-2], du


def make_dataset(snr=None, *, num_intervals=100, alpha=1.5, c=1.0, T=1.0, dt=0.01, noise_seed=0, amplitude=1.0):
    """Training pairs for FEX.

    amplitude scales the exact solution and the source together (the PDE is
    linear, so amplitude * u is still an exact solution). The paper's profile
    has |u| <= 0.3, where u, sin(u) and u + small*u^3 are nearly the same
    function; amplitude 5 gives |u| up to about 1.5 so FEX can tell them apart.
    The noise is set relative to the scaled signal, so the SNR is unchanged.

    Returns dict with
      states   (time, n_int + 2, 1)  observed u, ghosts fixed at 0
      targets  (time, n_int + 2, 1)  five-point u_t minus the known source f
      clean_targets                   exact u_t - f = -c (-Delta)^{alpha/2} u, for diagnostics
      adjacency, score_nodes, x, t
    """
    x, adjacency, score_nodes = build_graph(num_intervals, alpha)
    t = torch.arange(0.0, T + dt / 2, dt)
    u = amplitude * exact_solution(x, t, alpha)  # (time, n_int)

    observed = u.clone()
    if snr is not None:
        generator = torch.Generator().manual_seed(noise_seed + int(snr))
        observed = add_noise_db(u, snr, generator)

    u_mid, du = five_point(observed, dt)
    t_mid = t[2:-2]
    target = du - amplitude * source(x, t_mid, alpha, c)
    clean_target = -amplitude * c * fractional_laplacian_of_profile(x, alpha)[None, :] * torch.exp(-t_mid)[:, None]

    def with_ghosts(field):
        zeros = torch.zeros(field.shape[0], 2, dtype=field.dtype)
        return torch.cat([field, zeros], dim=1).unsqueeze(-1)

    return {
        "states": with_ghosts(u_mid),
        "targets": with_ghosts(target),
        "clean_targets": with_ghosts(clean_target),
        "adjacency": adjacency,
        "score_nodes": score_nodes,
        "x": x,
        "t": t_mid,
    }
