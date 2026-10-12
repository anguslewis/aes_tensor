"""S5: ties and floating-point precision.

Duplicated choices (several optimal menus), marginal values that are exactly
zero or within rounding of zero, one dominant choice next to a tiny one, and
profit scales from 1e-6 to 1e8; each with the default rtol and with rtol=0.
Also float32 correctness on problems whose profits are exact in float32.
"""
import pytest
import torch

import aes_tensor
import cdc_problems as P

N, C = 300, 8
RTOLS = (None, 0.0)


def solve_both(profit, marginal, data, params, mode, rtol):
    for m in (None, marginal):
        yield aes_tensor.solve(profit, data, params, n_choices=C, mode=mode, marginal=m, rtol=rtol)


@pytest.mark.parametrize("rtol", RTOLS)
@pytest.mark.parametrize("epsilon", (-0.3, 0.0, 0.5, 7.0))
def test_duplicated_choices(epsilon, rtol):
    data, params = P.make_ces(N, C, 5, epsilon, 0.8 if epsilon < 0 else 1.0)
    for name in ("phi", "cost"):
        data[name][:, 1] = data[name][:, 0]           # choices 0/1 and 2/3 are identical
        data[name][:, 3] = data[name][:, 2]
    value, menu, unique = P.brute_force(P.ces_profit, data, params, C)
    if epsilon > 0:
        # Substitutes take one of two identical choices, so menus tie. (With
        # complements or independence both copies go in or out together.)
        assert not unique.all()
    for sol in solve_both(P.ces_profit, P.ces_marginal, data, params, P.mode_for(epsilon), rtol):
        P.check(sol, P.ces_profit, data, params, value, menu, unique)


@pytest.mark.parametrize("rtol", RTOLS)
@pytest.mark.parametrize("sign", (1, -1))
def test_exact_zero_marginals(sign, rtol):
    # Small-integer pairwise problems: marginal values and menu values are
    # exact in float64, so many marginal values are exactly zero and many
    # menus tie exactly.
    data, params = P.make_pairwise(N, C, 6, sign, integer=True)
    value, menu, unique = P.brute_force(P.pairwise_profit, data, params, C)
    assert not unique.all()
    mode = "complements" if sign > 0 else "substitutes"
    for sol in solve_both(P.pairwise_profit, P.pairwise_marginal, data, params, mode, rtol):
        P.check(sol, P.pairwise_profit, data, params, value, menu, unique)


@pytest.mark.parametrize("rtol", RTOLS)
def test_independent_exact_zero_marginals(rtol):
    # epsilon = 0 with cost = K * phi exactly for half the choices: their
    # marginal value is zero up to the rounding of a sum.
    data, params = P.make_ces(N, C, 7, 0.0)
    data["cost"][:, ::2] = data["scale"][:, None] * data["phi"][:, ::2]
    value, menu, unique = P.brute_force(P.ces_profit, data, params, C)
    for mode in ("complements", "substitutes"):
        for sol in solve_both(P.ces_profit, P.ces_marginal, data, params, mode, rtol):
            P.check(sol, P.ces_profit, data, params, value, menu, unique)


@pytest.mark.parametrize("rtol", RTOLS)
@pytest.mark.parametrize("epsilon", (-0.3, 0.5, 7.0))
def test_near_zero_marginals(epsilon, rtol):
    # Each choice's cost equals its stand-alone value to within ~1e-15, so its
    # marginal value at the empty menu is within rounding of zero.
    data, params = P.make_ces(N, C, 8, epsilon)
    gen = torch.Generator().manual_seed(9)
    power = 1.0 / (1.0 + epsilon)
    standalone = data["scale"][:, None] * data["phi"].pow(power)
    data["cost"] = standalone * (1 + 1e-15 * torch.randn(N, C, generator=gen, dtype=P.DTYPE))
    value, menu, unique = P.brute_force(P.ces_profit, data, params, C)
    for sol in solve_both(P.ces_profit, P.ces_marginal, data, params, P.mode_for(epsilon), rtol):
        P.check(sol, P.ces_profit, data, params, value, menu, unique)


@pytest.mark.parametrize("rtol", RTOLS)
@pytest.mark.parametrize("epsilon", (-0.6, -0.3, 0.5, 2.0, 7.0))
def test_dominant_and_tiny_choice(epsilon, rtol):
    data, params = P.make_ces(N, C, 10, epsilon, 0.8 if epsilon < 0 else 1.0, dominant=True)
    value, menu, unique = P.brute_force(P.ces_profit, data, params, C)
    for sol in solve_both(P.ces_profit, P.ces_marginal, data, params, P.mode_for(epsilon), rtol):
        P.check(sol, P.ces_profit, data, params, value, menu, unique)


@pytest.mark.parametrize("rtol", RTOLS)
@pytest.mark.parametrize("scale", (1e-6, 1.0, 1e8))
@pytest.mark.parametrize("epsilon", (-0.3, 0.5, 7.0))
def test_profit_scale(epsilon, scale, rtol):
    data, params = P.make_ces(N, C, 11, epsilon, 0.8 if epsilon < 0 else 1.0, scale=scale)
    value, menu, unique = P.brute_force(P.ces_profit, data, params, C)
    for sol in solve_both(P.ces_profit, P.ces_marginal, data, params, P.mode_for(epsilon), rtol):
        P.check(sol, P.ces_profit, data, params, value, menu, unique)


@pytest.mark.parametrize("leaf_size", (10, 0))
@pytest.mark.parametrize("use_marginal", (False, True))
@pytest.mark.parametrize("sign", (1, -1))
def test_float32_exact_optimum(sign, use_marginal, leaf_size):
    # Small-integer pairwise problems: every profit is an integer, exact in
    # float32, so a unique optimum beats the next menu by at least 1. Keep the
    # agents whose optimum is unique; the solver must return exactly those
    # menus and profits in float32.
    data, params = P.make_pairwise(400, C, 60, sign, integer=True)
    value, menu, unique = P.brute_force(P.pairwise_profit, data, params, C)
    data32 = {k: v[unique].float() for k, v in data.items()}
    value, menu = value[unique], menu[unique]
    assert menu.any(1).any() and (value != 0).any()      # an all-empty, all-zero answer cannot pass
    mode = "complements" if sign > 0 else "substitutes"
    sol = aes_tensor.solve(P.pairwise_profit, data32, params, n_choices=C, mode=mode, dtype=torch.float32,
                           marginal=P.pairwise_marginal if use_marginal else None, leaf_size=leaf_size)
    assert sol.value.dtype == torch.float32
    assert torch.equal(sol.menu, menu)
    assert torch.equal(sol.value, value.float())
