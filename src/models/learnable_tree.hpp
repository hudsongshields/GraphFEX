#include <torch/torch.h>
#pragma once
#include "nodes.hpp"

struct LeafMLP : torch::nn::Module {
    torch::nn::Linear linear_ {nullptr};

    LeafMLP(int input_dim) {
        linear_ = register_module("linear_", torch::nn::Linear(input_dim, input_dim));

        torch::NoGradGuard no_grad;
        torch::nn::init::xavier_uniform_(linear_->weight);
        torch::nn::init::zeros_(linear_->bias);
    }
};

struct FEX : torch::nn::Module {
    torch::nn::ModuleList leafMLPs;
    Node parent_node;
};