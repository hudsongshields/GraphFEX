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
    parser.add_argument("--snr", type=int, default=None)
    parser.add_argument("--lambda_file",type=str, default=f"{TWOPHASEPATH}/thresholds/Lambda_HR.csv")
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

    Matrix = ElementaryFunctions_Matrix(
        TimeSeries,
        dim,
        n_nodes,
        A,
        self_poly_order,
        coupledPolyOrder=coupled_poly_order, TrigonometricIndex=False, CoupledTrigonometricIndex=False, 
        FractionalIndex=False, CoupledFractionalIndex=False, RescalingIndex=False, CoupledRescalingIndex=False,
    )
    Matrix = Matrix.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
    Matrix = Matrix.loc[:, ~Matrix.columns.str.contains("reg")]
    Matrix = Matrix.loc[:, ~Matrix.columns.str.contains("tanh", regex=False)]


    col_norms = Matrix.abs().sum(axis=0)
    Matrix = Matrix.loc[:, col_norms > col_norms.median() * 1e-6]

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


        selection_rate = (inferred.abs() > 1e-8).mean(axis=1)
        selected_terms = selection_rate[selection_rate >= 0.5].index.tolist()

        # The intercept is refit separately rather than treated as a library term.
        selected_terms = [term for term in selected_terms if term != "constant"]

        X_final = Matrix[selected_terms].to_numpy()
        y_final = NumDiv.iloc[:, d].to_numpy()

        if with_constant:
            X_final = np.column_stack([np.ones(X_final.shape[0]), X_final])
            coef, *_ = np.linalg.lstsq(X_final, y_final, rcond=None)

            final_fit = pd.Series(coef[1:], index=selected_terms, name="coefficient")
            final_fit.loc["constant"] = coef[0]

        else:
            coef, *_ = np.linalg.lstsq(X_final, y_final, rcond=None)
            final_fit = pd.Series(coef, index=selected_terms, name="coefficient")

        print(f"\nDimension {d + 1}")
        print(final_fit)

        final_fit.to_csv(f"{args.output}_dim{d + 1}_snr{args.snr}.csv")


if __name__ == "__main__":
    main()