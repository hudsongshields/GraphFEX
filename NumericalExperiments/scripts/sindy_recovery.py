import argparse
import importlib
import random

import numpy as np
import pysindy as ps
import torch

from NumericalExperiments.scindy_utils import CloneableCustomLibrary, GraphLibrary, to_numpy


def hr_data(snr: float, timesteps: int):
    from NumericalExperiments.HR.generate_data import make_static_sf_adjacency, make_timeseries

    adjacency = make_static_sf_adjacency(100, 500, gamma_in=3.5, gamma_out=3.5)
    states, derivatives = make_timeseries(
        num_samples=timesteps,
        adjacency=adjacency,
        snr=snr,
        smoothing=False,
    )
    return adjacency, states, derivatives


def stuart_landau_data(snr: float, timesteps: int):
    generator = importlib.import_module("NumericalExperiments.Stuart-Landau.generate_data")
    make_adjacency = generator.make_adjacency
    make_data = generator.make_data

    num_trajectories = 5
    num_timesteps = timesteps // num_trajectories

    adjacency = make_adjacency(100, 2, 7)
    states, derivatives, _ = make_data(
        num_trajectories=5,
        num_timesteps=num_timesteps,
        adjacency=adjacency,
        coupling=0.15,
        snr=snr,
        smoothing=False,
    )
    return adjacency, states, derivatives


def custom_library(functions, names):
    return CloneableCustomLibrary(library_functions=functions, function_names=names)


def hr_libraries():
    self_library = ps.PolynomialLibrary(degree=3, include_bias=True)
    neighbor_library = custom_library(
        [lambda value: 1 / (1 + np.exp(-value))],
        [lambda value: f"sigmoid({value})"],
    )
    return self_library, neighbor_library


def stuart_landau_libraries():
    self_library = ps.GeneralizedLibrary([
        ps.PolynomialLibrary(degree=3, include_bias=True),
        custom_library([lambda value: np.exp(-value)], [lambda value: f"exp(-{value})"]),
    ])
    neighbor_library = ps.GeneralizedLibrary([
        ps.PolynomialLibrary(degree=1, include_bias=True),
        custom_library([lambda value: np.exp(value)], [lambda value: f"exp({value})"]),
    ])
    return self_library, neighbor_library


SYSTEMS = {
    "hr": {
        "data_fn": hr_data,
        "library_fn": hr_libraries,
        "feature_names": ["x", "y", "z"],
        "output": "NumericalExperiments/HR/tests/sindy/recovered",
        "threshold": 0.5,
        "alpha": 0.25,
    },
    "stuart-landau": {
        "data_fn": stuart_landau_data,
        "library_fn": stuart_landau_libraries,
        "feature_names": ["x", "y"],
        "output": "NumericalExperiments/Stuart-Landau/tests/sindy/recovered",
        "threshold": 0.1,
        "alpha": 0.2,
    },
}


def fit_system(system_name: str, snr: float, timesteps: int, max_iter: int):
    system = SYSTEMS[system_name]
    adjacency, states, derivatives = system["data_fn"](snr, timesteps)
    self_library, neighbor_library = system["library_fn"]()
    feature_library = GraphLibrary(to_numpy(adjacency), self_library, neighbor_library)
    optimizer = ps.STLSQ(
        threshold=system["threshold"],
        alpha=system["alpha"],
        max_iter=max_iter,
        normalize_columns=True,
        unbias=True,
    )
    model = ps.SINDy(feature_library=feature_library, optimizer=optimizer)
    model.fit(to_numpy(states), t=0.01, x_dot=to_numpy(derivatives), feature_names=system["feature_names"])

    output_path = f"{system['output']}_{system_name}_snr{snr}.txt"
    with open(output_path, "w") as output_file:
        model.print(file=output_file)

    print(f"\n[{system_name}] snr={snr}")
    model.print()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems", nargs="+", choices=list(SYSTEMS), default=list(SYSTEMS))
    parser.add_argument("--snr-levels", type=float, nargs="+", default=[37.5, 45, 52.5, 60, 67.5])
    parser.add_argument("--timesteps", type=int, default=5000)
    parser.add_argument("--max-iter", type=int, default=20000)
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    for system_name in args.systems:
        for snr in args.snr_levels:
            fit_system(system_name, snr, args.timesteps, args.max_iter)


if __name__ == "__main__":
    main()