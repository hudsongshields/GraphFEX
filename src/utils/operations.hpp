#include <torch/torch.h>
#include <unordered_map>


namespace fexOperations {
    enum class Binary {
        add,
        sub,
        mul,
        div
    };
    enum class Unary {
        identity,
        square,
        cube,
        fourth_power, 
        exp,
        sigmoid,
        reciprocal
    };

    using BinaryOp = std::function<torch::Tensor(const torch::Tensor&, const torch::Tensor&)>;
    using UnaryOp = std::function<torch::Tensor(const torch::Tensor&)>;

    inline const std::unordered_map<Binary, BinaryOp> binaryMap{
        {Binary::add, [](const torch::Tensor& a, const torch::Tensor& b) {return torch::add(a, b); } },
        {Binary::sub, [](const torch::Tensor& a, const torch::Tensor& b) {return torch::sub(a, b); } },
        {Binary::mul, [](const torch::Tensor& a, const torch::Tensor& b) {return torch::mul(a, b); } },
        {Binary::div, [](const torch::Tensor& a, const torch::Tensor& b) {return torch::div(a, b); } },
    };
    inline const std::unordered_map<Unary, UnaryOp> unaryMap{
        {Unary::identity,     [](const torch::Tensor& x) {return x;} },
        {Unary::square,       [](const torch::Tensor& x) {return torch::pow(x, 2); } },
        {Unary::cube,         [](const torch::Tensor& x) {return torch::pow(x, 3);} },
        {Unary::fourth_power, [](const torch::Tensor& x) {return torch::pow(x, 4);} },
        {Unary::exp,          [](const torch::Tensor& x) {return torch::exp(x);} },
        {Unary::reciprocal,   [](const torch::Tensor& x) { return torch::reciprocal(x);} }
    };

    inline BinaryOp find_op(Binary op_name) {
        return binaryMap.at(op_name);
    };
    inline UnaryOp find_op(Unary op_name) {
        return unaryMap.at(op_name);
    };
};
