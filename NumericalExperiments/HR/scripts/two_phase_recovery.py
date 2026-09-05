import argparse
import random
import sys

import numpy as np
import pandas as pd
import torch

from NumericalExperiments.HR.generate_data import (make_static_sf_adjacency, make_timeseries)

TWOPHASEPATH = "NumericalExperiments/TwoPhase"
sys.path.append(TWOPHASEPATH)

from utils.ElementaryFunctions_Matrix import ElementaryFunctions_Matrix
from utils.TwoPhaseInference import TwoPhaseInference


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snr", type=int, default=60)
    parser.add_argument("--lambda_file",type=str, default=f"{TWOPHASEPATH}/Threshold/Lambda_HR.csv")
    parser.add_argument("--output", type=str, default="NumericalExperiments/HR/recovered_two_phase")
    args = parser.parse_args()

    timesteps = 5000
    n_nodes = 100
    dim = 3

    adj_matrix = make_static_sf_adjacency(n_nodes, 500, gamma_in=3.5, gamma_out=3.5)
    timeseries, t_derivs = make_timeseries(num_samples=timesteps, adjacency=adj_matrix, snr=args.snr)

    A = adj_matrix.cpu().numpy()

    n_samples = timeseries.shape[0]
    TimeSeries = timeseries.cpu().numpy().reshape(n_samples, -1)

    dX = t_derivs.cpu().numpy()  # [T, N, Dim]
    NumDiv = pd.DataFrame(dX.transpose(1, 0, 2).reshape(-1, dim))

    self_poly_order = 3
    coupled_poly_order = 1

    Matrix = ElementaryFunctions_Matrix(TimeSeries, dim, n_nodes, A, self_poly_order, coupledPolyOrder=coupled_poly_order)
    Matrix = Matrix.replace([np.inf, -np.inf], np.nan).dropna(axis=1)


    Lambda = pd.read_csv(args.lambda_file, header=None)

    keep = 10
    sample_times = 20
    batch_size = 10
    plot_start = 0.5
    plot_end = 0.7

    for d in range(dim):
        inferred, phase_one, waic, with_constant = TwoPhaseInference(
            Matrix, NumDiv,
            n_nodes, d, dim,
            keep,
            sample_times,
            batch_size,
            Lambda,
            plot_start, plot_end,
        )

        print(f"\nDimension {d + 1}")
        print(inferred)

        inferred.to_csv(f"{args.output}_dim{d + 1}_snr{args.snr}.csv")


if __name__ == "__main__":
    main()