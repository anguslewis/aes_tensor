"""S3: lower/upper restrictions.

Random forced and forbidden choices, agents with every choice fixed, agents
with nothing allowed, and padding columns. A profit function that returns NaN
for any menu outside an agent's bounds checks that the solver only evaluates
menus inside them.
"""
import pytest
import torch

import aes_tensor
import cdc_problems as P

N = 300


def random_bounds(n, c, seed, padding=2):
    gen = torch.Generator().manual_seed(1000 + seed)
    u = torch.rand(n, c, generator=gen)
    lower, upper = u < 0.15, u <= 0.8
    upper[0], lower[0] = lower[0], lower[0]          # agent 0: everything fixed
    upper[1], lower[1] = False, False                # agent 1: nothing allowed
    upper[:, c - padding:] = False                   # padding columns, unavailable to all
    lower[:, c - padding:] = False
    return lower, upper


def nan_outside(profit):
    """Wrap a profit so any menu outside the agent's [lower, upper] returns NaN."""
    def wrapped(menus, data, params):
        values = profit(menus, data, params)
        chosen = menus > 0.5
        outside = (data["lower"][:, None, :] & ~chosen).any(-1) | (chosen & ~data["upper"][:, None, :]).any(-1)
        return torch.where(outside, torch.full_like(values, torch.nan), values)
    return wrapped


CASES = [("ces", 0.5), ("ces", 7.0), ("ces", -0.3), ("ces", 0.0), ("pairwise", 1), ("pairwise", -1)]


@pytest.mark.parametrize("family, setting", CASES)
@pytest.mark.parametrize("c", (8, 12))
def test_restricted_matches_brute_force(family, setting, c):
    if family == "ces":
        data, params = P.make_ces(N, c, c, setting, 0.8 if setting < 0 else 1.0)
        profit, marginal, mode = P.ces_profit, P.ces_marginal, P.mode_for(setting)
    else:
        data, params = P.make_pairwise(N, c, c, setting)
        profit, marginal = P.pairwise_profit, P.pairwise_marginal
        mode = "complements" if setting > 0 else "substitutes"
    lower, upper = random_bounds(N, c, c)
    value, menu, unique = P.brute_force(profit, data, params, c, lower, upper)
    guarded = {**data, "lower": lower, "upper": upper}
    for use_marginal in (False, True):
        sol = aes_tensor.solve(nan_outside(profit), guarded, params, n_choices=c, mode=mode,
                               lower=lower, upper=upper, marginal=marginal if use_marginal else None)
        P.check(sol, profit, data, params, value, menu, unique, lower, upper)
    assert torch.equal(sol.menu[0], lower[0])
    assert not sol.menu[1].any() and float(sol.value[1]) == 0.0
