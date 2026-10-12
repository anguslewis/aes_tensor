"""S7: documented validation and numerical errors raise their exceptions.

(The squeezing loop's convergence cap is a safeguard only: bounds can only
tighten, so it cannot be reached and has no test.)
"""
import pytest
import torch

import aes_tensor
import cdc_problems as P

N, C = 20, 5
NON_FINITE = "Non-finite marginal value or tolerance encountered"


@pytest.fixture
def problem():
    return P.make_ces(N, C, 30, 1.0)


def run(data, params, profit=P.ces_profit, **kwargs):
    kwargs.setdefault("n_choices", C)
    kwargs.setdefault("mode", "substitutes")
    return aes_tensor.solve(profit, data, params, **kwargs)


def test_mode_is_required(problem):
    data, params = problem
    with pytest.raises(TypeError, match="mode"):
        aes_tensor.solve(P.ces_profit, data, params, n_choices=C)


def test_unknown_mode(problem):
    with pytest.raises(ValueError, match="mode must be one of"):
        run(*problem, mode="supermodular")


@pytest.mark.parametrize("kwargs, match", [
    ({"n_choices": 0}, "n_choices"),
    ({"n_choices": True}, "n_choices"),
    ({"leaf_size": -1}, "leaf_size"),
    ({"leaf_size": 1.5}, "leaf_size"),
    ({"rtol": -1.0}, "rtol"),
    ({"rtol": float("nan")}, "rtol"),
    ({"rtol": float("inf")}, "rtol"),
    ({"rtol": "small"}, "rtol"),
])
def test_bad_options(problem, kwargs, match):
    with pytest.raises(ValueError, match=match):
        run(*problem, **kwargs)


@pytest.mark.parametrize("dtype", (torch.int64, torch.float16, torch.bfloat16, "float64"))
def test_unsupported_dtype(problem, dtype):
    with pytest.raises(TypeError, match="dtype must be torch.float32 or torch.float64"):
        run(*problem, dtype=dtype)


@pytest.mark.parametrize("field, value, exc, match", [
    ("bad", 3.0, TypeError, "not a tensor"),
    ("bad", torch.tensor(1.0), ValueError, "0-d"),
    ("bad", torch.zeros(N + 1), ValueError, "rows along dimension 0"),
    ("bad", torch.zeros(0), ValueError, "rows along dimension 0"),
    ("bad", torch.ones(N, device="meta"), ValueError, "is on meta"),
])
def test_bad_data(problem, field, value, exc, match):
    data, params = problem
    with pytest.raises(exc, match=match):
        run({**data, field: value}, params)


def test_bad_container(problem):
    data, params = problem
    with pytest.raises(TypeError, match="dictionary of tensors"):
        run(list(data.values()), params)
    with pytest.raises(ValueError, match="at least one tensor"):
        run({}, params)


@pytest.mark.parametrize("first, second", [(3, 2), (2, 3), (3, 0)])
def test_none_key_does_not_skip_row_check(first, second):
    data = {None: torch.ones(first, 1, dtype=torch.float64), "x": torch.ones(second, 1, dtype=torch.float64)}
    with pytest.raises(ValueError, match="rows along dimension 0"):
        aes_tensor.solve(lambda m, d, p: m.sum(-1), data, None, n_choices=1, mode="substitutes")


def test_none_key_does_not_skip_device_check():
    data = {None: torch.ones(3, 1, dtype=torch.float64), "x": torch.ones(3, 1, device="meta")}
    with pytest.raises(ValueError, match="is on meta"):
        aes_tensor.solve(lambda m, d, p: m.sum(-1), data, None, n_choices=1, mode="substitutes")


@pytest.mark.parametrize("name", ("lower", "upper"))
def test_bad_bounds(problem, name):
    data, params = problem
    with pytest.raises(TypeError, match=f"{name} must be a bool tensor"):
        run(data, params, **{name: torch.zeros(N, C)})
    with pytest.raises(ValueError, match=f"{name} must have shape"):
        run(data, params, **{name: torch.zeros(N, C + 1, dtype=torch.bool)})
    with pytest.raises(ValueError, match=f"{name} is on meta"):
        run(data, params, **{name: torch.ones(N, C, dtype=torch.bool, device="meta")})


def test_lower_not_inside_upper(problem):
    data, params = problem
    lower = torch.zeros(N, C, dtype=torch.bool)
    lower[0, 0] = True
    upper = torch.ones(N, C, dtype=torch.bool)
    upper[0, 0] = False
    with pytest.raises(ValueError, match="subset of upper"):
        run(data, params, lower=lower, upper=upper)


def test_bad_profit_output(problem):
    data, params = problem
    with pytest.raises(ValueError, match="profit must return a tensor of shape"):
        run(data, params, profit=lambda m, d, p: P.ces_profit(m, d, p)[:, :1])
    with pytest.raises(ValueError, match="profit must return a tensor of shape"):
        run(data, params, profit=lambda m, d, p: None)
    with pytest.raises(TypeError, match="profit must return"):
        run(data, params, profit=lambda m, d, p: P.ces_profit(m, d, p).float())
    with pytest.raises(FloatingPointError, match="profit returned non-finite"):
        run(data, params, profit=lambda m, d, p: P.ces_profit(m, d, p) + torch.nan)
    with pytest.raises(FloatingPointError, match="profit returned non-finite"):
        run(data, params, profit=lambda m, d, p: P.ces_profit(m, d, p) + torch.inf)


def test_bad_marginal_output(problem):
    data, params = problem
    with pytest.raises(ValueError, match="D or \\(D, tol\\)"):
        run(data, params, marginal=lambda s, d, p: (*P.ces_marginal(s, d, p), None))
    with pytest.raises(ValueError, match="marginal D must have shape"):
        run(data, params, marginal=lambda s, d, p: P.ces_marginal(s, d, p)[0][:, :1])
    with pytest.raises(ValueError, match="marginal D must have shape"):
        run(data, params, marginal=lambda s, d, p: None)
    with pytest.raises(TypeError, match="marginal D must be"):
        run(data, params, marginal=lambda s, d, p: P.ces_marginal(s, d, p)[0].float())
    with pytest.raises(FloatingPointError, match="non-finite D"):
        run(data, params, marginal=lambda s, d, p: P.ces_marginal(s, d, p)[0] + torch.inf)
    with pytest.raises(FloatingPointError, match="non-finite D"):
        run(data, params, marginal=lambda s, d, p: P.ces_marginal(s, d, p)[0] + torch.nan)
    with pytest.raises(FloatingPointError, match="tol must be finite and non-negative"):
        run(data, params, marginal=lambda s, d, p: (P.ces_marginal(s, d, p)[0], -P.ces_marginal(s, d, p)[1] - 1))
    for bad in (torch.inf, torch.nan):
        with pytest.raises(FloatingPointError, match="tol must be finite and non-negative"):
            run(data, params, marginal=lambda s, d, p: (P.ces_marginal(s, d, p)[0], P.ces_marginal(s, d, p)[1] + bad))


# Finite profits whose differences or tolerances overflow near the dtype's
# limits: the solver must stop rather than rule a choice in or out on a NaN or
# an infinity (the one-choice optimum here is to include the choice).
@pytest.mark.parametrize("mode", ("complements", "substitutes"))
@pytest.mark.parametrize("rtol", (0.0, None))
@pytest.mark.parametrize("dtype, offset, slope", [(torch.float64, 1e308, 5e307), (torch.float32, 2e38, 1e38)])
def test_overflowing_tolerance_stops_the_solver(mode, rtol, dtype, offset, slope):
    profit = lambda m, d, p: offset + slope * m[..., 0]
    with pytest.raises(FloatingPointError, match=NON_FINITE):
        aes_tensor.solve(profit, {"id": torch.zeros(1)}, None, n_choices=1, mode=mode, rtol=rtol, dtype=dtype)


@pytest.mark.parametrize("mode", ("complements", "substitutes"))
def test_overflow_on_a_larger_additive_problem(mode):
    profit = lambda m, d, p: 1e308 + 1e305 * m.sum(-1)
    with pytest.raises(FloatingPointError, match=NON_FINITE):
        aes_tensor.solve(profit, {"id": torch.zeros(1)}, None, n_choices=12, mode=mode)


def test_oversized_rtol_overflows_in_float32():
    profit = lambda m, d, p: torch.zeros_like(m[..., 0])
    with pytest.raises(FloatingPointError, match=NON_FINITE):
        aes_tensor.solve(profit, {"id": torch.zeros(1)}, None, n_choices=2, mode="substitutes",
                         rtol=1e40, dtype=torch.float32)


def test_non_finite_unused_marginal_entries_are_ignored(problem):
    data, params = problem
    upper = torch.ones(N, C, dtype=torch.bool)
    upper[:, 0] = False                                   # choice 0 is never undecided

    def marginal(s, d, p):
        D, tol = P.ces_marginal(s, d, p)
        D[..., 0] = torch.nan
        tol[..., 0] = torch.inf
        return D, tol

    value, menu, unique = P.brute_force(P.ces_profit, data, params, C, upper=upper)
    sol = run(data, params, upper=upper, marginal=marginal)
    P.check(sol, P.ces_profit, data, params, value, menu, unique, upper=upper)
