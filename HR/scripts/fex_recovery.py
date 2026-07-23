import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random


from FEX.models import fex
from data.generate_data import make_data, make_static_sf_adjacency, make_timeseries
from FEX.utils.plots import plot_dynamics
from FEX.utils.metrics import sMAPE

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snr', type=float, default=40)
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    timesteps=5000
    adj_matrix = make_static_sf_adjacency(100, 500, gamma_in=3.5, gamma_out=3.5)
    timeseries, t_derivs = make_timeseries(num_samples=timesteps, adjacency=adj_matrix, snr=args.snr)


    dimx_fex = fex.CoupledFEX('depth_3_leaves_4_config', 'depth_2_tree_config', 0, controller_epochs=300, controller_lr=0.005, finetune_epochs=5000, finetune_lr=0.002,  num_fex_epochs=120, self_lr=0.02, inter_lr=0.02, bfgs_epochs=30, bfgs_lr=0.6, poolsize=8, device=device, expression_threshold=0.001)
    dimx_fex.fit(timeseries, t_derivs, adj_matrix, num_workers=5)

    dimy_fex = fex.SingleFEX('depth_2_tree_config', 1, num_finetune_epochs=3000, controller_epochs=200, num_fex_epochs=60, device=device, expression_threshold=0.001)
    dimy_fex.fit(timeseries, t_derivs, num_workers=5)

    dimz_fex = fex.SingleFEX('depth_2_tree_config', 2, controller_epochs=200, num_finetune_epochs=3000, num_fex_epochs=60, device=device, expression_threshold=0.001)
    dimz_fex.fit(timeseries, t_derivs, num_workers=5)

    predicted_states = torch.zeros(timesteps + 1, timeseries.size(1), timeseries.size(2), device='cpu')
    predicted_states[0] = timeseries[0]
    dt = 0.01
    with torch.no_grad():
        for t in range(timesteps):
            state = predicted_states[t]
            dx_dt = dimx_fex.predict(state, adj_matrix)
            dy_dt = dimy_fex.predict(state)
            dz_dt = dimz_fex.predict(state)
            deriv = torch.cat([
                dx_dt.cpu(),
                dy_dt.cpu(),
                dz_dt.cpu()
            ], dim=-1)

            predicted_states[t+1] = state + dt * deriv

            if not torch.isfinite(predicted_states[t+1]).all():
                print(f"Non-finite state at timestep {t + 1}")
                break

    node = 10
    fig = plot_dynamics(timeseries[:, node, 0].cpu(), timeseries[:, node, 1].cpu(), timeseries[:, node, 2].cpu(), predicted_states[:, node, :].cpu(), elev=15, azim=75)
    fig.savefig(f"node_{node}_dynamics_snr{args.snr}.png")
    plt.close(fig)

    print("Final FEX expressions")
    print(dimx_fex)
    print(dimy_fex)
    print(dimz_fex)
    ground_truth_self_str = "-1.0*x1**3 + 3.0*x1**2 + 1.0*x2 - 1.0*x3 + 3.24"
    ground_truth_inter_str = "(0.3 - 0.15*x1)*exp(x4)/(exp(x4) + 1)"
    smape_self = sMAPE(ground_truth_self_str, str(dimx_fex.best_model.forcing_tree))
    smape_inter = sMAPE(ground_truth_inter_str, str(dimx_fex.best_model.inter_tree))
    print("sMAPE for self dynamics:", smape_self)
    print("sMAPE for inter dynamics:", smape_inter)
    print("Total sMAPE:", smape_self + smape_inter)

    # if there is a checkpoint, instead of just printing, save to a txt file
    if args.checkpoint is not None:
        with open(args.checkpoint, 'w') as f:
            f.write("Final FEX expressions\n")
            f.write(str(dimx_fex) + "\n")
            f.write(str(dimy_fex) + "\n")
            f.write(str(dimz_fex) + "\n")
            f.write("sMAPE for self dynamics: " + str(smape_self) + "\n")
            f.write("sMAPE for inter dynamics: " + str(smape_inter) + "\n")
            f.write("Total sMAPE: " + str(smape_self + smape_inter) + "\n")

if __name__ == "__main__":
    main()