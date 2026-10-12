"""S6: the interface contract.

The dictionary-only data container, mode aliases, agent permutation, user
batching, params passed through untouched, degenerate sizes, and the stats
counters.
"""
import dataclasses
import types
from collections import OrderedDict
from typing import NamedTuple

import pytest
import torch

import aes_tensor
import cdc_problems as P

N, C = 400, 8


@pytest.fixture
def problem():
    data, params = P.make_ces(N, C, 20, 1.0)
    value, menu, unique = P.brute_force(P.ces_profit, data, params, C)
    return data, params, value, menu, unique


def solve(data, params, **kwargs):
    kwargs.setdefault("mode", "substitutes")
    return aes_tensor.solve(P.ces_profit, data, params, n_choices=C, **kwargs)


class CesTuple(NamedTuple):
    phi: torch.Tensor
    scale: torch.Tensor
    cost: torch.Tensor


@dataclasses.dataclass
class Doubling:
    """Non-idempotent initializer: rebuilding the object would change its data."""
    weight: torch.Tensor

    def __post_init__(self):
        self.weight = self.weight * 2


@dataclasses.dataclass
class NeedsInitVar:
    weight: torch.Tensor
    seed: dataclasses.InitVar[int]

    def __post_init__(self, seed):
        pass


@dataclasses.dataclass(frozen=True)
class Frozen:
    weight: torch.Tensor


@dataclasses.dataclass(slots=True)
class Slotted:
    weight: torch.Tensor


def weight_profit(menus, data, params):
    return (menus * (data.weight[:, None, :] - 0.75)).sum(2)


def test_only_dictionaries_are_accepted(problem):
    data, params, *_ = problem
    weight = torch.tensor([[0.25]], dtype=torch.float64)
    doubling = Doubling(weight)
    stored = doubling.weight.clone()
    rejected = [doubling, NeedsInitVar(weight, seed=1), Frozen(weight), Slotted(weight)]
    for container in rejected:
        with pytest.raises(TypeError, match="dictionary of tensors"):
            aes_tensor.solve(weight_profit, container, None, n_choices=1, mode="substitutes")
    assert torch.equal(doubling.weight, stored)          # nothing re-ran the initializer
    for container in (CesTuple(**data), types.MappingProxyType(data), list(data.values())):
        with pytest.raises(TypeError, match="dictionary of tensors"):
            solve(container, params)


def test_dict_subclasses_are_dictionaries(problem):
    data, params, *_ = problem
    reference = solve(data, params)
    sol = solve(OrderedDict(data), params)
    assert torch.equal(sol.menu, reference.menu) and torch.equal(sol.value, reference.value)


def test_mode_aliases():
    for epsilon, mode, alias in ((1.0, "substitutes", "scdc_above"), (-0.3, "complements", "scdc_below")):
        data, params = P.make_ces(N, C, 21, epsilon, 0.8 if epsilon < 0 else 1.0)
        a, b = solve(data, params, mode=mode), solve(data, params, mode=alias)
        assert torch.equal(a.menu, b.menu) and torch.equal(a.value, b.value)


def test_permuting_agents_permutes_results(problem):
    data, params, value, menu, unique = problem
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(22))
    sol = solve({k: v[perm] for k, v in data.items()}, params)
    P.check(sol, P.ces_profit, {k: v[perm] for k, v in data.items()}, params,
            value[perm], menu[perm], unique[perm])


def test_user_batching_equals_one_call(problem):
    data, params, *_ = problem
    whole = solve(data, params)
    parts = [solve({k: v[i:j] for k, v in data.items()}, params) for i, j in ((0, 150), (150, 151), (151, N))]
    assert torch.equal(torch.cat([p.menu for p in parts]), whole.menu)
    assert P.close(torch.cat([p.value for p in parts]), whole.value).all()


def test_params_passed_through_untouched(problem):
    data, params, *_ = problem
    seen = []

    def spy_profit(menus, d, p):
        seen.append(p is params)
        return P.ces_profit(menus, d, p)

    def spy_marginal(sets, d, p):
        seen.append(p is params)
        return P.ces_marginal(sets, d, p)

    for marginal in (None, spy_marginal):
        aes_tensor.solve(spy_profit, data, params, n_choices=C, mode="substitutes", marginal=marginal)
    assert seen and all(seen)


def test_degenerate_sizes():
    data, params = P.make_ces(5, C, 23, 1.0)
    empty = solve({k: v[:0] for k, v in data.items()}, params)
    assert empty.menu.shape == (0, C) and empty.value.shape == (0,)
    for n, c in ((1, C), (5, 1), (1, 1)):
        data, params = P.make_ces(n, c, 24, 1.0)
        value, menu, unique = P.brute_force(P.ces_profit, data, params, c)
        sol = aes_tensor.solve(P.ces_profit, data, params, n_choices=c, mode="substitutes")
        P.check(sol, P.ces_profit, data, params, value, menu, unique)


def test_stats(problem):
    data, params, *_ = problem
    a = solve(data, params)
    b = solve(data, params, marginal=P.ces_marginal)
    keys = {"squeeze_iterations", "max_unresolved", "branched_rows", "leaf_menus", "profit_calls", "marginal_calls"}
    assert set(a.stats) == keys and set(b.stats) == keys
    assert a.stats["profit_calls"] > 0 and a.stats["marginal_calls"] == 0
    assert b.stats["marginal_calls"] > 0 and a.stats["leaf_menus"] >= N
