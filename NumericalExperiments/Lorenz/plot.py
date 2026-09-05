import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

seed = 42

def reseed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

reseed(seed)


from .generate_data import (make_adjacency, make_data)

timesteps = 5000
dt = 0.01

adj_matrix = make_adjacency(100, 3).cpu()

print("Mean degree:", torch.mean(adj_matrix.sum(dim=1)))


# Reseed before every trajectory so that the initial conditions
reseed(seed)
snr_30_t, _ = make_data(
    num_samples=timesteps,
    adjacency=adj_matrix,
    snr=30,
    coupling=0.8,
)

reseed(seed)
snr_45_t, _ = make_data(
    num_samples=timesteps,
    adjacency=adj_matrix,
    snr=45,
    coupling=0.8,
)

reseed(seed)
snr_60_t, _ = make_data(
    num_samples=timesteps,
    adjacency=adj_matrix,
    snr=60,
    coupling=0.8,
)

reseed(seed)
timeseries, _ = make_data(
    num_samples=timesteps,
    adjacency=adj_matrix,
    snr=None,
    coupling=0.8,
)

snr_30_t = snr_30_t.cpu()
snr_45_t = snr_45_t.cpu()
snr_60_t = snr_60_t.cpu()
timeseries = timeseries.cpu()


# Learned Lorenz equations
def dimx_fex_predict(state, adj_matrix, snr=None):
    x = state[:, 0]
    y = state[:, 1]

    # Number of incoming neighbors for each node
    degree = adj_matrix.sum(dim=1)

    """    
    if snr is None:
        interaction = (
            0.8 * (adj_matrix @ x)
            - 0.7999 * degree * x
        )

        dx_dt = (
            10.00034322 * y
            - 10.00034322 * x
            + interaction
        )"""
    if snr is None:
        interaction = (
            0.8 * (adj_matrix @ x)
            - 0.8 * degree * x
        )

        dx_dt = (
            10.0 * y
            - 10.0 * x
            + interaction
        )
    elif snr == 60:
        # dx/dt =
        # -9.9998 x + 9.9999 y
        # + sum_j A_ij (0.8 x_j - 0.7998 x_i)

        interaction = (
            0.8 * (adj_matrix @ x)
            - 0.7998 * degree * x
        )

        dx_dt = (
            -9.9998 * x
            + 9.9999 * y
            + interaction
        )

    elif snr == 45:
        # dx/dt =
        # 10.0011 y - 10.00022 x
        # + sum_j A_ij (0.7984 x_j - 0.7965 x_i)

        interaction = (
            0.7984 * (adj_matrix @ x)
            - 0.7965 * degree * x
        )

        dx_dt = (
            10.0011 * y
            - 10.00022 * x
            + interaction
        )

    elif snr == 30:
        interaction = (
            0.74190909 * (adj_matrix @ x)
            - 0.7268608 * degree * x
        )

        dx_dt = (
            9.9761 * y
            - 9.946 * x
            + interaction
        )

    else:
        raise ValueError(f"Unsupported SNR: {snr}")

    return dx_dt.unsqueeze(-1)


def dimy_fex_predict(state, snr=None):
    x = state[:, 0]
    y = state[:, 1]
    z = state[:, 2]
    """
    if snr is None:
        # dy/dt =
        # 27.9883414 x - 1.00168564 xz - 0.9376185 y

        dy_dt = (
            27.9883414 * x
            - 1.00168564 * x * z
            - 0.9376185 * y
        )"""
    if snr is None:
        dy_dt = (
            28.0 * x
            - 1.0 * x * z
            - 1.0 * y
        )
    elif snr == 60:
        dy_dt = (
            27.95826985 * x
            - 1.0001916 * x * z
            - 0.99310104 * y
        )

    elif snr == 45:

        dy_dt = (
            27.96935776 * x
            - 0.99736584 * x * z
            - 0.993 * y
        )

    elif snr == 30:
        dy_dt = (27.73889817 * x - 0.99576372 * x * z - 0.91021066 * y)

    else:
        raise ValueError(f"Unsupported SNR: {snr}")

    return dy_dt.unsqueeze(-1)


def dimz_fex_predict(state, snr=None):
    x = state[:, 0]
    y = state[:, 1]
    z = state[:, 2]
    """
    if snr is None:
        # dz/dt = 0.99974122 xy - 2.66662728 z
        dz_dt = (0.99974122 * x * y - 2.66662728 * z)
    """
    if snr is None:
        dz_dt = (x * y - (8/3) * z)
    elif snr == 60:
        dz_dt = (0.99999082 * x * y - 2.6669586 * z)

    elif snr == 45:
        dz_dt = (1.0000368 * x * y - 2.6667981 * z)

    elif snr == 30:
        dz_dt = (0.99536178 * x * y - 2.69228723 * z)

    else:
        raise ValueError(f"Unsupported SNR: {snr}")

    return dz_dt.unsqueeze(-1)


def rk4_step(state, dt, adj_matrix, snr=None):

    k1 = torch.cat([
        dimx_fex_predict(state, adj_matrix, snr=snr),
        dimy_fex_predict(state, snr=snr),
        dimz_fex_predict(state, snr=snr),
    ], dim=-1)

    state2 = state + 0.5 * dt * k1

    k2 = torch.cat([
        dimx_fex_predict(state2, adj_matrix, snr=snr),
        dimy_fex_predict(state2, snr=snr),
        dimz_fex_predict(state2, snr=snr),
    ], dim=-1)

    state3 = state + 0.5 * dt * k2

    k3 = torch.cat([
        dimx_fex_predict(state3, adj_matrix, snr=snr),
        dimy_fex_predict(state3, snr=snr),
        dimz_fex_predict(state3, snr=snr),
    ], dim=-1)

    state4 = state + dt * k3

    k4 = torch.cat([
        dimx_fex_predict(state4, adj_matrix, snr=snr),
        dimy_fex_predict(state4, snr=snr),
        dimz_fex_predict(state4, snr=snr),
    ], dim=-1)

    return state + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0


def init_states(timesteps, timeseries):
    predicted_states = torch.zeros(
        timesteps + 1,
        timeseries.size(1),
        timeseries.size(2),
    )
    predicted_states[0] = timeseries[0]


    snr_30_pred = torch.zeros(
        timesteps + 1,
        timeseries.size(1),
        timeseries.size(2),
    )
    snr_30_pred[0] = timeseries[0]


    snr_45_pred = torch.zeros(
        timesteps + 1,
        timeseries.size(1),
        timeseries.size(2),
    )
    snr_45_pred[0] = timeseries[0]


    snr_60_pred = torch.zeros(
        timesteps + 1,
        timeseries.size(1),
        timeseries.size(2),
    )
    snr_60_pred[0] = timeseries[0]

    return predicted_states, snr_30_pred, snr_45_pred, snr_60_pred


# Integrate learned systems
def integrate_trajectories(predicted_states, snr_30_pred, snr_45_pred, snr_60_pred, timesteps, dt, adj_matrix):
    with torch.no_grad():
        for t in range(timesteps):

            predicted_states[t + 1] = rk4_step(
                predicted_states[t],
                dt,
                adj_matrix,
                snr=None,
            )

            snr_30_pred[t + 1] = rk4_step(
                snr_30_pred[t],
                dt,
                adj_matrix,
                snr=30,
            )

            snr_45_pred[t + 1] = rk4_step(
                snr_45_pred[t],
                dt,
                adj_matrix,
                snr=45,
            )

            snr_60_pred[t + 1] = rk4_step(
                snr_60_pred[t],
                dt,
                adj_matrix,
                snr=60,
            )

from FEX.helpers.plots import plot_panel
def main():
    predicted_states, snr_30_pred, snr_45_pred, snr_60_pred = init_states(timesteps, timeseries)
    integrate_trajectories(predicted_states, snr_30_pred, snr_45_pred, snr_60_pred, timesteps, dt, adj_matrix)

    node = 80

    _ = plot_panel(
        nonePred=predicted_states[:, node, :].cpu(),
        noneTrue=timeseries[:, node, :].cpu(),

        snr60Pred=snr_60_pred[:, node, :].cpu(),
        snr60True=snr_60_t[:, node, :].cpu(),

        snr45Pred=snr_45_pred[:, node, :].cpu(),
        snr45True=snr_45_t[:, node, :].cpu(),

        snr30Pred=snr_30_pred[:, node, :].cpu(),
        snr30True=snr_30_t[:, node, :].cpu(),

        elev=30,
        azim=180,
        save_path="NumericalExperiments/Lorenz/test_panel_dynamics"
    )