import argparse
import random
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd
import torch

TWOPHASEPATH = "NumericalExperiments/TwoPhase"
sys.path.append(TWOPHASEPATH)

from utils.ElementaryFunctions_Matrix import ElementaryFunctions_Matrix
from utils.TwoPhaseInference import TwoPhaseInference

from FEX.helpers.metrics import sMAPE

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# ElementaryFunctionsPool coupled columns are named "xNj"/"xNixNj"/"xNjMinusxNi" (and the
# sigmoid-activation equivalents). "N" indexes the self node's variable (x1..x3); the neighbor's
# equivalent variable is written as x{N+3}, following FEX's edge-variable convention.
_COUPLED_PATTERNS = [
    (re.compile(r"^x(\d)j$"), lambda n: f"x{n + 3}"),
    (re.compile(r"^x(\d)ix\1j$"), lambda n: f"x{n}*x{n + 3}"),
    (re.compile(r"^x(\d)jMinusx\1i$"), lambda n: f"x{n + 3} - x{n}"),
    (re.compile(r"^sigx(\d)jalpha1beta0$"), lambda n: f"exp(x{n + 3})/(1 + exp(x{n + 3}))"),
    (re.compile(r"^sigx(\d)ix\1jalpha1beta0$"), lambda n: f"x{n}*exp(x{n + 3})/(1 + exp(x{n + 3}))"),
    (re.compile(r"^sigx(\d)jMinusx\1ialpha1beta0$"), lambda n: f"exp(x{n + 3} - x{n})/(1 + exp(x{n + 3} - x{n}))"),
    (re.compile(r"^x(\d)isigx\1jalpha1beta0$"), lambda n: f"x{n}*exp(x{n + 3})/(1 + exp(x{n + 3}))"),
]


def self_term_to_sympy(term: str) -> str:
    """Translate an ElementaryFunctionsPool self-polynomial column (e.g. 'x1x1x2') into a monomial."""
    factors = re.findall(r"x\d+", term)
    counts = Counter(factors)
    return "*".join(f"{var}**{count}" if count > 1 else var for var, count in sorted(counts.items()))


def term_to_sympy(term: str) -> tuple[bool, str]:
    """Translate a library column name into a sympy-parseable expression, tagged coupled vs. self."""
    for pattern, to_expr in _COUPLED_PATTERNS:
        match = pattern.match(term)
        if match:
            return True, to_expr(int(match.group(1)))
    return False, self_term_to_sympy(term)


def fit_to_sympy_strs(final_fit: pd.Series) -> tuple[str, str]:
    """Split a recovered coefficient series into (self_expr, inter_expr) sympy-parseable strings."""
    self_parts = []
    inter_parts = []
    for term, coef in final_fit.items():
        if term == "constant":
            self_parts.append(f"({coef})")
            continue
        is_coupled, expr = term_to_sympy(term)
        part = f"({coef})*({expr})"
        (inter_parts if is_coupled else self_parts).append(part)

    self_str = " + ".join(self_parts) if self_parts else "0"
    inter_str = " + ".join(inter_parts) if inter_parts else "0"
    return self_str, inter_str


def hr_data(snr):
    from NumericalExperiments.HR.generate_data import make_static_sf_adjacency, make_timeseries

    adj_matrix = make_static_sf_adjacency(100, 500, gamma_in=3.5, gamma_out=3.5)
    timeseries, t_derivs = make_timeseries(num_samples=5000, adjacency=adj_matrix, snr=snr, smoothing=False)
    return adj_matrix, timeseries, t_derivs


def lorenz_data(snr):
    from NumericalExperiments.Lorenz.generate_data import make_adjacency, make_data

    adj_matrix = make_adjacency(100, 3)
    timeseries, t_derivs = make_data(num_samples=5000, adjacency=adj_matrix, snr=snr, coupling=0.8, smoothing=False)
    return adj_matrix, timeseries, t_derivs


SYSTEMS = {
    "hr": dict(
        data_fn=hr_data,
        self_poly_order=3,
        coupled_poly_order=1,
        library_kwargs=dict(
            TrigonometricIndex=False, ExponentialIndex=False, FractionalIndex=False, ActivationIndex=False,
            RescalingIndex=False, CoupledTrigonometricIndex=False, CoupledExponentialIndex=False,
            CoupledFractionalIndex=False, CoupledRescalingIndex=False,
        ),
        lambda_file=f"{TWOPHASEPATH}/thresholds/Lambda_HR.csv",
        output=f"{TWOPHASEPATH}/../HR/recovered_two_phase",
        ground_truth={
            0: ("-1.0*x1**3 + 3.0*x1**2 + 1.0*x2 - 1.0*x3 + 3.24", "(0.3 - 0.15*x1)*exp(x4)/(exp(x4) + 1)"),
            1: ("1.0 - 5.0*x1**2 - 1.0*x2", "0"),
            2: ("0.02*x1 + 0.032 - 0.005*x3", "0"),
        },
    ),
    "lorenz": dict(
        data_fn=lorenz_data,
        self_poly_order=2,
        coupled_poly_order=1,
        library_kwargs=dict(
            TrigonometricIndex=False, ExponentialIndex=False, FractionalIndex=False, ActivationIndex=False,
            RescalingIndex=False, CoupledTrigonometricIndex=False, CoupledExponentialIndex=False,
            CoupledFractionalIndex=False, CoupledActivationIndex=False, CoupledRescalingIndex=False,
        ),
        lambda_file=f"{TWOPHASEPATH}/thresholds/Lambda_Lorenz.csv",
        output=f"{TWOPHASEPATH}/../Lorenz/recovered_two_phase",
        ground_truth={
            0: ("-10.0*x1 + 10.0*x2", "-0.8*x1 + 0.8*x4"),
            1: ("28.0*x1 - 1.0*x1*x3 - 1.0*x2", "0"),
            2: ("1.0*x1*x2 - 2.6666666667*x3", "0"),
        },
    ),
}


def build_matrix(system, adj_matrix, timeseries, t_derivs):
    dim = 3
    A = adj_matrix.cpu().numpy()
    n_samples = timeseries.shape[0]
    TimeSeries = timeseries.cpu().numpy().reshape(n_samples, -1)
    dX = t_derivs.cpu().numpy()
    NumDiv = pd.DataFrame(dX.transpose(1, 0, 2).reshape(-1, dim))

    Matrix = ElementaryFunctions_Matrix(
        TimeSeries,
        dim,
        adj_matrix.shape[0],
        A,
        system["self_poly_order"],
        coupledPolyOrder=system["coupled_poly_order"], **system["library_kwargs"],
    )
    Matrix = Matrix.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
    Matrix = Matrix.loc[:, ~Matrix.columns.str.contains("reg")]
    Matrix = Matrix.loc[:, ~Matrix.columns.str.contains("tanh", regex=False)]

    col_norms = Matrix.abs().sum(axis=0)
    Matrix = Matrix.loc[:, col_norms > col_norms.median() * 1e-6]

    return Matrix, NumDiv


def run_system(system_name, snr):
    system = SYSTEMS[system_name]

    adj_matrix, timeseries, t_derivs = system["data_fn"](snr)
    Matrix, NumDiv = build_matrix(system, adj_matrix, timeseries, t_derivs)
    Lambda = pd.read_csv(system["lambda_file"], header=None)

    n_nodes = adj_matrix.shape[0]
    dim = 3
    keep = 10
    sample_times = 30
    batch_size = 10
    plot_start = 0.5
    plot_end = 0.7

    # Only the coupled dimension (d == 0) is worth recovering; dims 2/3 are not scored.
    d = 0
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

    print(f"\n[{system_name}] snr={snr} Dimension {d + 1}")
    print(final_fit)

    pred_self_str, pred_inter_str = fit_to_sympy_strs(final_fit)
    true_self_str, true_inter_str = system["ground_truth"][d]

    smape_self = sMAPE(true_self_str, pred_self_str)
    smape_inter = sMAPE(true_inter_str, pred_inter_str)
    print(f"sMAPE self: {smape_self:.4f}  sMAPE inter: {smape_inter:.4f}")
    coupled_dim_smape = (smape_self + smape_inter) / 2

    smape_series = pd.Series(
        {"sMAPE_self": smape_self, "sMAPE_inter": smape_inter, "sMAPE_total": coupled_dim_smape},
        name="coefficient",
    )
    pd.concat([final_fit, smape_series]).to_csv(f"{system['output']}_{system_name}_dim{d + 1}_snr{snr}.csv")

    print(f"\n[{system_name}] snr={snr} coupled-dim sMAPE: {coupled_dim_smape:.4f}")
    return coupled_dim_smape


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems", nargs="+", choices=list(SYSTEMS), default=list(SYSTEMS))
    parser.add_argument("--snr_levels", type=int, nargs="+", default=[30, 35, 40, 45, 50, 55, 60])
    args = parser.parse_args()

    results = {}
    for system_name in args.systems:
        for snr in args.snr_levels:
            results[(system_name, snr)] = run_system(system_name, snr)

    print("\nSummary (coupled-dim sMAPE):")
    for (system_name, snr), coupled_dim_smape in results.items():
        print(f"  {system_name} snr={snr}: {coupled_dim_smape:.4f}")


if __name__ == "__main__":
    main()
