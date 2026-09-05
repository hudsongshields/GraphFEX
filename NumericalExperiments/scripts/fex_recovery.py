import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from FEX.utils import fex
from FEX.helpers.plots import plot_dynamics


def hr_data(args):
    from NumericalExperiments.HR.generate_data import make_static_sf_adjacency, make_timeseries

    adj_matrix = make_static_sf_adjacency(100, 500, gamma_in=3.5, gamma_out=3.5)
    timeseries, t_derivs = make_timeseries(num_samples=args.timesteps, adjacency=adj_matrix, snr=args.snr)
    return adj_matrix, timeseries, t_derivs


def hr_fex_specs(device):
    return {
        0: ("coupled", dict(
            self_fex_struct="depth_3_leaves_4_config", inter_fex_struct="depth_2_tree_config", target_dim=0,
            controller_epochs=400, controller_lr=0.003, finetune_epochs=20000, finetune_lr=1e-4,
            num_fex_epochs=120, self_lr=0.02, inter_lr=0.02, bfgs_epochs=0, bfgs_lr=0.65, poolsize=10,
            device=device, expression_threshold=0.1,
        ), dict(num_workers=5, finetune_bs=64)),
        1: ("single", dict(
            self_fex_struct="depth_2_tree_config", target_dim=1, num_finetune_epochs=5000,
            controller_epochs=200, num_fex_epochs=60, device=device, expression_threshold=0.01,
        ), dict(num_workers=5)),
        2: ("single", dict(
            self_fex_struct="depth_2_tree_config", target_dim=2, controller_lr=0.01, controller_epochs=250,
            num_finetune_epochs=10000, finetune_lr=0.002, self_lr=0.04, num_fex_epochs=80, device=device,
            expression_threshold=0.001,
        ), dict(num_workers=5)),
    }


def lorenz_data(args):
    from NumericalExperiments.Lorenz.generate_data import make_adjacency, make_data

    adj_matrix = make_adjacency(100, 3)
    timeseries, t_derivs = make_data(num_samples=args.timesteps, adjacency=adj_matrix, snr=args.snr, coupling=0.8, smoothing=False)
    return adj_matrix, timeseries, t_derivs


def lorenz_fex_specs(device):
    return {
        0: ("coupled", dict(
            self_fex_struct="depth_2_tree_config", inter_fex_struct="depth_2_tree_config", target_dim=0,
            controller_epochs=200, controller_lr=0.005, finetune_epochs=20000, finetune_lr=1e-4,
            num_fex_epochs=120, self_lr=0.02, inter_lr=0.02, bfgs_epochs=20, bfgs_lr=1.0, poolsize=8,
            device=device, expression_threshold=0.1,
        ), dict(num_workers=5, finetune_bs=512)),
        1: ("single", dict(
            self_fex_struct="depth_3_leaves_4_config", target_dim=1, num_finetune_epochs=10000,
            controller_epochs=200, num_fex_epochs=80, finetune_lr=0.002, device=device,
        ), dict(num_workers=5)),
        2: ("single", dict(
            self_fex_struct="depth_3_leaves_4_config", target_dim=2, num_finetune_epochs=10000,
            num_fex_epochs=100, finetune_lr=4e-4, device=device,
        ), dict(num_workers=5)),
    }


def kuramoto_data(args):
    from NumericalExperiments.Kuramoto.generate_data import make_arni_adjacency, make_data

    adj_matrix = make_arni_adjacency(100, 5)
    timeseries, t_derivs = make_data(num_samples=args.timesteps, adjacency=adj_matrix, snr=args.snr, coupling=1.0)
    return adj_matrix, timeseries, t_derivs


def kuramoto_fex_specs(device):
    return {
        0: ("coupled", dict(
            self_fex_struct="depth_2_tree_config", inter_fex_struct="depth_2_tree_config", target_dim=0,
            controller_epochs=100, inter_lr=0.02, self_lr=0.02, num_fex_epochs=80, finetune_epochs=5000,
            finetune_lr=1e-4, poolsize=8, device=device, expression_threshold=0.01,
        ), dict(num_workers=5)),
    }


DATA_FNS = {"hr": hr_data, "lorenz": lorenz_data, "kuramoto": kuramoto_data}
SPEC_FNS = {"hr": hr_fex_specs, "lorenz": lorenz_fex_specs, "kuramoto": kuramoto_fex_specs}


def build_and_fit(kind, model_kwargs, fit_kwargs, timeseries, t_derivs, adj_matrix):
    model_kwargs = dict(model_kwargs)
    if kind == "coupled":
        model = fex.CoupledFEX(
            model_kwargs.pop("self_fex_struct"), model_kwargs.pop("inter_fex_struct"),
            model_kwargs.pop("target_dim"), **model_kwargs,
        )
        model.fit(timeseries, t_derivs, adj_matrix, **fit_kwargs)
    else:
        model = fex.SingleFEX(model_kwargs.pop("self_fex_struct"), model_kwargs.pop("target_dim"), **model_kwargs)
        model.fit(timeseries, t_derivs, **fit_kwargs)
    return model



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["hr", "lorenz", "kuramoto"], required=True)
    parser.add_argument("--snr", type=float, default=None)
    parser.add_argument("--timesteps", type=int, default=5000)
    parser.add_argument("--target_dim", type=int, default=None, help="Only train a single dimension (default: all)")
    args = parser.parse_args()

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    adj_matrix, timeseries, t_derivs = DATA_FNS[args.experiment](args)
    fex_specs = SPEC_FNS[args.experiment](device)

    dims = [args.target_dim] if args.target_dim is not None else sorted(fex_specs)

    models = {}
    for dim in dims:
        kind, model_kwargs, fit_kwargs = fex_specs[dim]
        models[dim] = build_and_fit(kind, model_kwargs, fit_kwargs, timeseries, t_derivs, adj_matrix)
        print(f"Dim {dim}: {models[dim]}")



if __name__ == "__main__":
    main()
