import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
def reseed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

from data.generate_data import make_static_sf_adjacency
timesteps=5000
adj_matrix = make_static_sf_adjacency(100, 500, gamma_in=3.5, gamma_out=3.5)
print(torch.mean(adj_matrix.sum(dim=1))) # number of incoming edges (degree)
from data.generate_data import make_timeseries
snr_30_t, _ = make_timeseries(num_samples=5000, adjacency=adj_matrix, snr=30)
reseed(seed)
snr_45_t, _ = make_timeseries(num_samples=5000, adjacency=adj_matrix, snr=45)
reseed(seed)
snr_60_t, _ = make_timeseries(num_samples=5000, adjacency=adj_matrix, snr=60)
reseed(seed)
timeseries, _ = make_timeseries(num_samples=5000, adjacency=adj_matrix, snr=None)


def dimx_fex_predict(state, adj_matrix, snr=None):
    x = state[:, 0]
    y = state[:, 1]
    z = state[:, 2]
    if not snr:
        interaction = (0.3000816 - 0.14999958*x) * (adj_matrix @ torch.sigmoid(x))
        dx_dt = y - x**3 + 3*x**2 - 0.9999*z + 3.24 + interaction
    elif snr == 60:
        interaction = (0.30008918 - 0.14999106*x) * (adj_matrix @ torch.sigmoid(x))
        dx_dt = y - 0.9999*x**3 + 2.9998*x**2 - 1.0001*z + 3.24 + interaction
    elif snr == 45:
        interaction = 0.15 * (2.0 - x) * (adj_matrix @ torch.sigmoid(x))
        dx_dt = 1.0007*y - 0.9964*x**3 + 2.9932*x**2 - 0.9976*z + 3.2384 + interaction
    elif snr == 30:# TODO: has to be updated once recovered correctly
        interaction = 0.15 * (2.0 - x) * (adj_matrix @ torch.sigmoid(x)) 
        dx_dt = y - x**3 + 3*x**2 - z + 3.24 + interaction
    return dx_dt.unsqueeze(-1)
def dimy_fex_predict(state, snr=None):
    x = state[:, 0]
    y = state[:, 1]
    if not snr:
        dy_dt = 1 - 5 * x**2 - y
    elif snr == 60:
        dy_dt = 1.0022 - 4.9996*x**2 - 0.9996*y
    elif snr == 45:
        dy_dt = 1.0052 - 4.9923*x**2 - 0.999*y
    elif snr == 30:
        dy_dt = 1.0634 - 4.9185*x**2 - 0.986*y
    return dy_dt.unsqueeze(-1)
def dimz_fex_predict(state, snr=None):
    x = state[:, 0]
    z = state[:, 2]
    if not snr:
        dz_dt = 0.02 * x - 0.005 * z + 0.032
    elif snr == 60:
        dz_dt = 0.0205942*x - 0.0051667*z + 0.03279584
    elif snr == 45:
        dz_dt = 0.0183233*x - 0.00439483*z + 0.0306147
    elif snr == 30:
        dz_dt = 0.02 * x - 0.005 * z + 0.032 # TODO: has to be updated once recovered correctly
    return dz_dt.unsqueeze(-1)

def rk4_step(state, dt, adj_matrix, snr=None):
    k1 = torch.cat([
        dimx_fex_predict(state, adj_matrix, snr=snr), 
        dimy_fex_predict(state, snr=snr),
        dimz_fex_predict(state, snr=snr)
    ], dim=-1)
    k2 = torch.cat([
        dimx_fex_predict(state + 0.5 * dt * k1, adj_matrix, snr=snr), 
        dimy_fex_predict(state + 0.5 * dt * k1, snr=snr),
        dimz_fex_predict(state + 0.5 * dt * k1, snr=snr)
    ], dim=-1)
    k3 = torch.cat([
        dimx_fex_predict(state + 0.5 * dt * k2, adj_matrix, snr=snr), 
        dimy_fex_predict(state + 0.5 * dt * k2, snr=snr),
        dimz_fex_predict(state + 0.5 * dt * k2, snr=snr)
    ], dim=-1)
    k4 = torch.cat([
        dimx_fex_predict(state + dt * k3, adj_matrix, snr=snr), 
        dimy_fex_predict(state + dt * k3, snr=snr),
        dimz_fex_predict(state + dt * k3, snr=snr)
    ], dim=-1)
    return state + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0



predicted_states = torch.zeros(timesteps + 1, timeseries.size(1), timeseries.size(2), device='cpu')
predicted_states[0] = timeseries[0]

snr_30_pred = torch.zeros(timesteps + 1, snr_30_t.size(1), snr_30_t.size(2), device='cpu')
snr_30_pred[0] = snr_30_t[0]
snr_45_pred = torch.zeros(timesteps + 1, snr_45_t.size(1), snr_45_t.size(2), device='cpu')
snr_45_pred[0] = snr_45_t[0]
snr_60_pred = torch.zeros(timesteps + 1, snr_60_t.size(1), snr_60_t.size(2), device='cpu')
snr_60_pred[0] = snr_60_t[0]
dt = 0.01
with torch.no_grad():
    for t in range(timesteps):
        predicted_states[t+1] = rk4_step(predicted_states[t], dt, adj_matrix.to('cpu'), snr=None)
        snr_30_pred[t+1] = rk4_step(snr_30_pred[t], dt, adj_matrix.to('cpu'), snr=30)
        snr_45_pred[t+1] = rk4_step(snr_45_pred[t], dt, adj_matrix.to('cpu'), snr=45)
        snr_60_pred[t+1] = rk4_step(snr_60_pred[t], dt, adj_matrix.to('cpu'), snr=60)


        
from FEX.utils.plots import plot_panel

node = 80
fig = plot_panel(
    nonePred=predicted_states[:, node, :].cpu(),
    noneTrue=timeseries[:, node, :].cpu(),
    snr60Pred=snr_60_pred[:, node, :].cpu(),
    snr60True=snr_60_t[:, node, :].cpu(),
    snr45Pred=snr_45_pred[:, node, :].cpu(),
    snr45True=snr_45_t[:, node, :].cpu(),
    snr30Pred=snr_30_pred[:, node, :].cpu(),
    snr30True=snr_30_t[:, node, :].cpu(),
    elev=30,
    azim=45
)
plt.show()
