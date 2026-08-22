import argparse
import torch
import pysindy as ps
import numpy as np 
import pandas as pd 
import random 
seed = 42 
random.seed(seed) 
np.random.seed(seed) 
torch.manual_seed(seed) 
if torch.cuda.is_available(): 
    torch.cuda.manual_seed_all(seed) 
from FEX.models import fex 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from NumericalExperiments.HR.data.generate_data import make_static_sf_adjacency
import matplotlib.pyplot as plt
from NumericalExperiments.HR.data.generate_data import make_timeseries



from ...scindy_utils import GraphLibrary, CloneableCustomLibrary, to_numpy
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snr', type=int, default=60)
    parser.add_argument('--output', type=str, default="NumericalExperiments/HR/recovered_scindy")
    args = parser.parse_args()

    timesteps=5000
    dt = 0.01
    adj_matrix = make_static_sf_adjacency(100, 500, gamma_in=3.5, gamma_out=3.5)
    timeseries, t_derivs = make_timeseries(num_samples=timesteps, adjacency=adj_matrix, snr=args.snr)

    sigmoid_functions = [lambda x: 1 / (1 + np.exp(-x))]
    sigmoid_names = [lambda x: f"sigmoid({x})"]
    sigmoid_library = CloneableCustomLibrary(library_functions=sigmoid_functions, function_names=sigmoid_names)

    self_library = (ps.PolynomialLibrary(degree=3, include_bias=True) + sigmoid_library)
    neighbor_library = (ps.PolynomialLibrary(degree=3, include_bias=False) + sigmoid_library)
    graph_library = GraphLibrary(adjacency=to_numpy(adj_matrix), self_library=self_library, neighbor_library=neighbor_library)

    n_iter = 20000
    optimizer = ps.STLSQ(threshold=0.1, alpha=0.01, max_iter=n_iter, unbias=True, normalize_columns=False)

    model = ps.SINDy(feature_library=graph_library, optimizer=optimizer)
    model.fit(to_numpy(timeseries), t=dt, x_dot=to_numpy(t_derivs), feature_names=["x", "y", "z"])
    model.print()
    if args.output is not None:
        log_path = f"{args.output}_snr{args.snr}.txt"
        with open(log_path, "w") as f:
            model.print(file=f)
            f.write(f"STLSQ iterations: {n_iter}/{model.optimizer.max_iter}\n")

if __name__ == "__main__":
    main()
