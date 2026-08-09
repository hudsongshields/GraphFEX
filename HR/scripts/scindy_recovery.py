import argparse

import torch 
import torch 
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
from data.generate_data import make_static_sf_adjacency
import matplotlib.pyplot as plt
from data.generate_data import make_timeseries


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()

    return np.asarray(x)

from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted
from pysindy.feature_library.base import BaseFeatureLibrary, x_sequence_or_item
from pysindy.utils import AxesArray
class GraphLibrary(BaseFeatureLibrary):
    def __init__(self, adjacency, self_library, neighbor_library):
        self.adjacency = adjacency
        self.self_library = self_library
        self.neighbor_library = neighbor_library

    @x_sequence_or_item
    def fit(self, x_full, y=None):
        arrays = [np.asarray(x) for x in x_full]

        num_dimensions = arrays[0].shape[2]
        self.adjacency_ = to_numpy(self.adjacency)

        x_flat = np.concatenate([x.reshape(-1, num_dimensions) for x in arrays], axis=0)

        self.self_library_ = clone(self.self_library).fit(x_flat)
        self.neighbor_library_ = clone(self.neighbor_library).fit(x_flat)
        self.n_features_in_ = num_dimensions

        num_self = self.self_library_.n_output_features_
        num_neighbor = self.neighbor_library_.n_output_features_
        self.n_output_features_ = num_self + num_self * num_neighbor

        return self

    @x_sequence_or_item
    def transform(self, x_full):
        check_is_fitted(self)
        transformed = []

        for x in x_full:
            x_array = np.asarray(x)
            num_times, num_nodes, num_dimensions = x_array.shape
            x_flat = x_array.reshape(num_times * num_nodes, num_dimensions)

            theta_self = np.asarray(self.self_library_.transform(x_flat)).reshape(num_times, num_nodes, -1)

            theta_neighbor = np.asarray(self.neighbor_library_.transform(x_flat)).reshape(num_times, num_nodes, -1)
            theta_graph = self.adjacency_[None, :, :] @ theta_neighbor
            theta_interaction = (theta_self[..., :, None] * theta_graph[..., None, :]).reshape(num_times, num_nodes, -1)

            theta_full = np.concatenate([theta_self, theta_interaction], axis=-1)

            transformed.append(AxesArray(theta_full, x.axes))

        return transformed

    def get_feature_names(self, input_features=None):
        check_is_fitted(self)

        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]

        self_names = list(self.self_library_.get_feature_names(input_features))
        neighbor_names = list(self.neighbor_library_.get_feature_names(input_features))
        graph_names = [rf"\sum_j A_{{ij}}({name}_j)" for name in neighbor_names]

        interaction_names = [f"({self_name})({graph_name})" for self_name in self_names for graph_name in graph_names]
        return self_names + interaction_names

import pysindy as ps


class CloneableCustomLibrary(ps.CustomLibrary):
    def __init__(self, library_functions, function_names=None, interaction_only=True, include_bias=False):
        self.library_functions = library_functions
        super().__init__(
            library_functions=library_functions,
            function_names=function_names,
            interaction_only=interaction_only,
            include_bias=include_bias,
        )

from sklearn.linear_model import Lasso
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snr', type=int, default=60)
    parser.add_argument('--output', type=str, default="recovered_scindy")
    args = parser.parse_args()

    timesteps=5000
    dt = 0.01
    adj_matrix = make_static_sf_adjacency(100, 500, gamma_in=3.5, gamma_out=3.5)
    timeseries, t_derivs = make_timeseries(num_samples=timesteps, adjacency=adj_matrix, snr=args.snr)

    sigmoid_functions = [lambda x: 1 / (1 + np.exp(-x))]
    sigmoid_names = [lambda x: f"sigmoid({x})"]
    sigmoid_library = CloneableCustomLibrary(library_functions=sigmoid_functions, function_names=sigmoid_names)

    self_library = (ps.PolynomialLibrary(degree=3, include_bias=True) + ps.FourierLibrary(n_frequencies=1) + sigmoid_library)
    neighbor_library = (ps.PolynomialLibrary(degree=3, include_bias=False) + ps.FourierLibrary(n_frequencies=1) + sigmoid_library)
    graph_library = GraphLibrary(adjacency=to_numpy(adj_matrix), self_library=self_library, neighbor_library=neighbor_library)

    n_iter = 20000
    optimizer = ps.STLSQ(threshold=0.1, alpha=0.55, max_iter=n_iter)

    model = ps.SINDy(feature_library=graph_library, optimizer=optimizer)
    model.fit(to_numpy(timeseries), t=dt, x_dot=to_numpy(t_derivs), feature_names=["x", "y", "z"])
    model.print()
    if args.output is not None:
        log_path = f"{args.output}_snr{args.snr}.txt"
        with open(log_path, "w") as f:
            model.print(file=f)
            f.write(f"SR3 iterations: {n_iter}/{model.optimizer.max_iter}\n")

if __name__ == "__main__":
    main()
