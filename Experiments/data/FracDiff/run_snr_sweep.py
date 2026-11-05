"""Small FEX search on the fPINNs 1D space-fractional example, SNR 30-70 dB.

Time derivative: five-point differences (no smoothing). Target: u_t - f.
Truth: forcing F(u_i) = 0, interaction G(u_i, u_j) = c * (u_j - u_i) with c = 1.
"""

import argparse
import json
from pathlib import Path
import math
import random

import numpy as np
import sympy as sp
import torch

from FEX.utils.fex import CoupledFEX
from Experiments.data.FracDiff.generate_data import make_dataset

TRUE_C = 1.0


def reseed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def interaction_coefficients(model):
    """Coefficients of u_i, u_j and the constant in the interaction tree."""
    ui, uj = sp.symbols("ui uj")
    expr = model.best_model.inter_tree.symbolic_expression(["ui", "uj"])
    # Slopes at u = 0, so a near-linear choice such as sin(u) ~ u still gives c.
    # |u| <= 0.3 in this example, where sin, identity and tanh-like ops barely differ.
    at_zero = {ui: 0, uj: 0}
    return {
        "ui": float(sp.diff(expr, ui).subs(at_zero)),
        "uj": float(sp.diff(expr, uj).subs(at_zero)),
        "const": float(expr.subs(at_zero)),
        "expr": str(expr),
    }



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snrs", default="none,30,37.5,45,52.5,60,67.5")
    parser.add_argument("--self-tree", default="depth_1_tree_config")
    parser.add_argument("--inter-tree", default="depth_1_tree_config")
    parser.add_argument("--controller-epochs", type=int, default=16)
    parser.add_argument("--cands", type=int, default=8)
    parser.add_argument("--fex-epochs", type=int, default=20)
    # This problem is badly conditioned (edge weights ~ h^-alpha, and u_j and
    # u_i enter almost collinearly), so Adam alone stalls far from the optimum.
    # L-BFGS reaches the least-squares solution; give it enough iterations.
    parser.add_argument("--bfgs-epochs", type=int, default=50)
    parser.add_argument("--bfgs-lr", type=float, default=1.0)
    # CoupledFEX.fit fine-tunes with Adam only and keeps the best *Adam* epoch,
    # which can be worse than the L-BFGS solution it started from. 0 = just
    # re-score the pool on the held-out split.
    parser.add_argument("--finetune-epochs", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-intervals", type=int, default=100)
    parser.add_argument("--amplitude", type=float, default=1.0)
    parser.add_argument("--out", default="fracdiff_results.json")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    out_path = script_dir / args.out

    snrs = [None if s.lower() == "none" else float(s) for s in args.snrs.split(",")]
    results = []
    for snr in snrs:
        reseed(args.seed)
        data = make_dataset(snr, num_intervals=args.num_intervals, amplitude=args.amplitude)
        model = CoupledFEX(
            args.self_tree, args.inter_tree, target_dim=0,
            controller_epochs=args.controller_epochs, cands_per_cycle=args.cands,
            num_fex_epochs=args.fex_epochs, bfgs_epochs=args.bfgs_epochs,
            self_lr=0.05, inter_lr=0.05, bfgs_lr=args.bfgs_lr,
            finetune_epochs=args.finetune_epochs, finetune_lr=1e-2,
            poolsize=3,
        )
        model.fit(
            data["states"], data["targets"], data["adjacency"],
            batch_size=32, num_workers=args.workers, finetune_bs=32,
            score_nodes=data["score_nodes"],
        )
        coeffs = interaction_coefficients(model)
        forcing = str(model.best_model.forcing_tree.symbolic_expression(["u"]))
        # c is read from the u_j coefficient. The u_i coefficient is not identifiable
        # on its own: every interior node has the same total edge weight D, so
        # w_ui * D * u_i trades off exactly against a linear forcing term a * u.
        c_hat = coeffs["uj"]
        row = {
            "snr": snr,
            "num_intervals": args.num_intervals,
            "amplitude": args.amplitude,
            "forcing": forcing,
            "interaction": coeffs["expr"],
            "c_hat": c_hat,
            "c_rel_err": abs(c_hat - TRUE_C) / TRUE_C,
            "ui_coeff": coeffs["ui"],
            "reward": model.best_model.reward,
            "inter_ops": [int(i) for i in model.best_model.inter_tree.sample_indices],
            "self_ops": [int(i) for i in model.best_model.forcing_tree.sample_indices],
        }
        results.append(row)
        print("RESULT", json.dumps(row), flush=True)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
