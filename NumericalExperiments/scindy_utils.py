import numpy as np
import torch
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted
from pysindy.feature_library.base import BaseFeatureLibrary, x_sequence_or_item
from pysindy.utils import AxesArray
import pysindy as ps


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()

    return np.asarray(x)

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


class CloneableCustomLibrary(ps.CustomLibrary):
    def __init__(self, library_functions, function_names=None, interaction_only=True, include_bias=False):
        self.library_functions = library_functions
        super().__init__(
            library_functions=library_functions,
            function_names=function_names,
            interaction_only=interaction_only,
            include_bias=include_bias,
        )