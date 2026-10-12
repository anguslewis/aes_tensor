"""S4: large C, where brute force is impossible.

C in {27, 40} (27 is the size of our currency application), 2,000 agents,
costs drawn with the realistic calibration. A must equal B, and the result
must not depend on leaf_size: menus exactly, values to the pass-rule profit
tolerance.
"""
import pytest
import torch

import aes_tensor
import cdc_problems as P

N = 2000
CASES = [("ces", eps) for eps in (-0.6, -0.3, 0.5, 7.0)] + [("pairwise", 1), ("pairwise", -1)]


@pytest.mark.parametrize("family, setting", CASES)
@pytest.mark.parametrize("c", (27, 40))
def test_a_equals_b_and_leaf_size_invariance(family, setting, c):
    if family == "ces":
        data, params = P.make_ces(N, c, c, setting, realistic=True)
        profit, marginal, mode = P.ces_profit, P.ces_marginal, P.mode_for(setting)
    else:
        data, params = P.make_pairwise(N, c, c, setting, strength=0.05, density=0.2)
        profit, marginal = P.pairwise_profit, P.pairwise_marginal
        mode = "complements" if setting > 0 else "substitutes"
    reference = aes_tensor.solve(profit, data, params, n_choices=c, mode=mode)
    runs = [aes_tensor.solve(profit, data, params, n_choices=c, mode=mode, marginal=marginal)]
    runs += [aes_tensor.solve(profit, data, params, n_choices=c, mode=mode, leaf_size=k) for k in (0, 2)]
    assert reference.stats["max_unresolved"] > 0
    for sol in [reference] + runs:
        P.check(sol, profit, data, params, reference.value, reference.menu, torch.ones(N, dtype=torch.bool))
