import sympy as sp

def coefficient_map(expr: str) -> dict:
    expr = sp.expand(sp.sympify(expr))
    coefficients = {}

    for term in sp.Add.make_args(expr):
        coefficient, basis = term.as_coeff_Mul()
        coefficients[basis] = coefficients.get(basis, 0) + coefficient

    return coefficients

def sMAPE(true_expr: str, pred_expr: str) -> float:
    true_coeffs = coefficient_map(true_expr)
    pred_coeffs = coefficient_map(pred_expr)
    all_terms = true_coeffs.keys() | pred_coeffs.keys()

    if not all_terms:
        return 0.0

    print("All terms:", all_terms)
    smape_total = 0.0
    for term in all_terms:
        true_val = float(true_coeffs.get(term, 0))
        pred_val = float(pred_coeffs.get(term, 0))
        denominator = abs(true_val) + abs(pred_val)

        if denominator != 0:
            smape_total += abs(pred_val - true_val) / denominator

    return 100.0 * smape_total / len(all_terms)