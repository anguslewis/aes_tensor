"""S1: the solver finds the brute-force optimum.

CES and pairwise objectives over C in {1, 2, 3, 5, 8, 12}, solved both with
profit only (A) and with a closed-form marginal (B).
"""
import pytest
import torch

import aes_tensor
import cdc_problems as P

SIZES = (1, 2, 3, 5, 8, 12)
N = 300
EPSILONS = (-0.95, -0.6, -0.3, 0.0, 0.5, 2.0, 7.0, 50.0)
SPREADS = {"mild": (0.3, False), "wide": (2.0, False), "dominant": (1.0, True)}


def cost_levels(epsilon):
    # Complementary objectives are near all-or-nothing, so their interior is narrower.
    return (0.6, 0.8, 1.0) if epsilon < 0 else (0.5, 1.0, 2.0)


CES_CASES = [(eps, level, spread, mode)
             for eps in EPSILONS
             for level in cost_levels(eps)
             for spread in SPREADS
             for mode in (("complements", "substitutes") if eps == 0 else (P.mode_for(eps),))]


@pytest.mark.parametrize("epsilon, cost_level, spread, mode", CES_CASES)
def test_ces_matches_brute_force(epsilon, cost_level, spread, mode):
    phi_spread, dominant = SPREADS[spread]
    for seed, c in enumerate(SIZES):
        data, params = P.make_ces(N, c, seed, epsilon, cost_level, phi_spread, dominant)
        value, menu, unique = P.brute_force(P.ces_profit, data, params, c)
        for marginal in (None, P.ces_marginal):
            sol = aes_tensor.solve(P.ces_profit, data, params, n_choices=c, mode=mode, marginal=marginal)
            P.check(sol, P.ces_profit, data, params, value, menu, unique)


PAIRWISE_CASES = [(sign, strength, density)
                  for sign in (1, -1) for strength in (0.1, 1.0, 3.0) for density in (1.0, 0.3)]


@pytest.mark.parametrize("sign, strength, density", PAIRWISE_CASES)
def test_pairwise_matches_brute_force(sign, strength, density):
    mode = "complements" if sign > 0 else "substitutes"
    for seed, c in enumerate(SIZES):
        data, params = P.make_pairwise(N, c, seed, sign, strength, density)
        value, menu, unique = P.brute_force(P.pairwise_profit, data, params, c)
        for marginal in (None, P.pairwise_marginal):
            sol = aes_tensor.solve(P.pairwise_profit, data, params, n_choices=c, mode=mode, marginal=marginal)
            P.check(sol, P.pairwise_profit, data, params, value, menu, unique)


def test_oracle_rejects_a_wrong_optimum():
    # One agent, two choices, additive: a = [1, -1e15], W = 0. The unique
    # optimum is {choice 0} with profit 1. The extreme loss on choice 1 must
    # not loosen the pass rule enough to accept the empty menu with profit 0.
    data = {"a": torch.tensor([[1.0, -1e15]], dtype=P.DTYPE), "W": torch.zeros(1, 2, 2, dtype=P.DTYPE)}
    value, menu, unique = P.brute_force(P.pairwise_profit, data, {}, 2)
    assert value.tolist() == [1.0] and menu.tolist() == [[True, False]] and unique.all()
    wrong = aes_tensor.Solution(torch.zeros(1, 2, dtype=torch.bool), torch.zeros(1, dtype=P.DTYPE), {})
    with pytest.raises(AssertionError):
        P.check(wrong, P.pairwise_profit, data, {}, value, menu, unique)
    for mode in ("complements", "substitutes"):
        sol = aes_tensor.solve(P.pairwise_profit, data, {}, n_choices=2, mode=mode)
        P.check(sol, P.pairwise_profit, data, {}, value, menu, unique)
