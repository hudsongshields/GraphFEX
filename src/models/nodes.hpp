#include <torch/torch.h>

#pragma once
#include "utils/operations.hpp"

struct BinaryOp : torch::nn::Module {
    fexOperations::BinaryOp op_;

    BinaryOp(fexOperations::Binary op) : op_(fexOperations::find_op(op)){}
    torch::Tensor forward(torch::Tensor a, torch::Tensor b){
        return op_(a, b);
    }
};

struct UnaryOp : torch::nn::Module {
    torch::Tensor a_;
    torch::Tensor b_;
    fexOperations::UnaryOp op_;

    UnaryOp(fexOperations::Unary op) 
    : op_{fexOperations::find_op(op)}
    {
        auto weight_tensor = torch::empty({1});
        auto bias_tensor = torch::empty({1});

        {
            torch::NoGradGuard no_grad;
            torch::nn::init::normal_(weight_tensor, 0.0, 2.0);
            torch::nn::init::zeros_(bias_tensor);
        }

        a_ = register_parameter("a_", weight_tensor);
        b_ = register_parameter("b_", bias_tensor);
    }

    torch::Tensor forward(torch::Tensor x) {
        return op_(x);
    }
};

class Node {
    std::shared_ptr<torch::nn::Module> operation_;
    std::unique_ptr<Node> left{nullptr};
    std::unique_ptr<Node> right{nullptr};
    void init();

    public:
        Node(fexOperations::Binary op_name)
        : operation_{std::make_shared<BinaryOp>(op_name)}
        {}

        Node(fexOperations::Unary op_name)
        : operation_{std::make_shared<UnaryOp>(op_name)}
        {}

        torch::Tensor operator()();

        bool is_leaf() {
            if (left==nullptr && right==nullptr) return true;
            else return false;
        };

};