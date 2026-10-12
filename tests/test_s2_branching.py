"""S2: branching and leaf size.

The same problems solved with leaf_size in {0, 1, 2, 10, C} must all reach
the brute-force optimum. Near-symmetric choices defeat squeezing, so those
cases branch even at leaf_size = 10; leaf_size = C never branches.
"""
import pytest

import aes_tensor
import cdc_problems as P

N, C = 300, 12
LEAF_SIZES = (0, 1, 2, 10, C)

CASES = [
    ("ces", -0.6, False), ("ces", 0.5, False), ("ces", 7.0, False),
    ("ces", 1.0, True), ("ces", -0.5, True),
    ("pairwise", 1, False), ("pairwise", -1, False),
]


def build(family, setting, near_symmetric):
    if family == "ces":
        data, params = P.make_ces(N, C, 50, setting, 0.8 if setting < 0 else 1.0, near_symmetric=near_symmetric)
        return data, params, P.ces_profit, P.ces_marginal, P.mode_for(setting)
    data, params = P.make_pairwise(N, C, 50, setting, strength=3.0)
    return data, params, P.pairwise_profit, P.pairwise_marginal, ("complements" if setting > 0 else "substitutes")


@pytest.mark.parametrize("family, setting, near_symmetric", CASES)
def test_every_leaf_size_reaches_the_optimum(family, setting, near_symmetric):
    data, params, profit, marginal, mode = build(family, setting, near_symmetric)
    value, menu, unique = P.brute_force(profit, data, params, C)
    for leaf_size in LEAF_SIZES:
        for m in (None, marginal):
            sol = aes_tensor.solve(profit, data, params, n_choices=C, mode=mode, marginal=m, leaf_size=leaf_size)
            P.check(sol, profit, data, params, value, menu, unique)
            unresolved = sol.stats["max_unresolved"]
            if leaf_size >= C:
                assert sol.stats["branched_rows"] == 0
            elif unresolved > leaf_size:
                assert sol.stats["branched_rows"] > 0
            if near_symmetric:
                # squeezing cannot separate (almost) identical choices
                assert unresolved == C and sol.stats["branched_rows"] > 0 or leaf_size >= C
