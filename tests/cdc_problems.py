"""Test problems, the brute-force reference, and the pass rule.

Two objective families, each satisfying the single-crossing condition for its
mode by construction:

* CES aggregator: pi(S) = K * (sum_{c in S} phi_c) ** p - sum_{c in S} cost_c,
  p = 1 / (1 + epsilon). epsilon > 0 is submodular (substitutes), -1 < epsilon < 0
  supermodular (complements), epsilon = 0 additively separable (both modes).
* Pairwise interactions (Jia 2008 style): pi(S) = sum_{j in S} a_j
  + 1/2 sum_{j != k in S} W_jk with W symmetric, zero diagonal. W >= 0 is
  supermodular (complements), W <= 0 submodular (substitutes).

Pass rule: menus must equal the reference exactly wherever the reference
optimum is unique, and profit values must satisfy
abs(candidate - reference) <= 1e-8 + 1e-8 * abs(reference). Menus whose
profit is within that tolerance of the best are treated as tied, and any
tied menu is accepted. The reported value must also equal the profit of the
reported menu, and every value must be finite.
"""
import torch

DTYPE = torch.float64
PROFIT_ATOL = 1e-8
PROFIT_RTOL = 1e-8


def mode_for(epsilon):
    """The mode under which the CES objective is valid (epsilon = 0 is valid in both)."""
    return "substitutes" if epsilon > 0 else "complements"


# --------------------------------------------------------------------------
# CES aggregator
# --------------------------------------------------------------------------
def ces_profit(menus, data, params):
    power = 1.0 / (1.0 + params["epsilon"])
    phi_sum = torch.einsum("bmc,bc->bm", menus, data["phi"])
    return data["scale"][:, None] * phi_sum.pow(power) - torch.einsum("bmc,bc->bm", menus, data["cost"])


def ces_marginal(sets, data, params):
    """Closed-form marginal values (approach B), with a rounding bound."""
    power = 1.0 / (1.0 + params["epsilon"])
    phi = data["phi"][:, None, :]
    selected = sets * phi
    zero = torch.zeros_like(selected[..., :1])
    before = torch.cat((zero, selected[..., :-1].cumsum(-1)), -1)
    after = torch.cat((selected[..., 1:].flip(-1).cumsum(-1).flip(-1), zero), -1)
    off = before + after
    off_power, on_power = off.pow(power), (off + phi).pow(power)
    scale = data["scale"][:, None, None]
    cost = data["cost"][:, None, :]
    D = scale * (on_power - off_power) - cost
    unit = 64 * torch.finfo(D.dtype).eps
    tol = unit * scale.abs() * (on_power.abs() + off_power.abs()) + unit * cost.abs()
    return D, tol


def make_ces(n, c, seed, epsilon, cost_level=1.0, phi_spread=1.0, dominant=False, scale=1.0,
             near_symmetric=False, realistic=False):
    """A CES problem with interior solutions.

    Each choice's cost is cost_level x (its marginal value when added to a menu
    holding half the total phi) x U(0.5, 1.5), so cost_level ~ 1 puts choices
    near the margin for any epsilon. That is adversarial for squeezing at large
    C (every choice sits at the margin), so realistic=True instead spreads each
    cost in logs around the interval between the choice's marginal value at the
    empty and the full menu: most choices are clearly not worth taking, a few
    clearly are, some are in between (cost_level is then unused).
    dominant=True makes choice 0 a million times larger and choice 1 a million
    times smaller than the rest. near_symmetric gives (almost) identical choices
    with a common cost, which defeats squeezing and forces branching.
    """
    gen = torch.Generator().manual_seed(seed)
    if near_symmetric:
        phi = 1.0 + 1e-3 * torch.rand(n, c, generator=gen, dtype=DTYPE)
    else:
        phi = torch.exp(phi_spread * torch.randn(n, c, generator=gen, dtype=DTYPE))
    if dominant and c >= 2:
        phi[:, 0] *= 1e6
        phi[:, 1] *= 1e-6
    K = scale * torch.exp(0.5 * torch.randn(n, generator=gen, dtype=DTYPE))
    power = 1.0 / (1.0 + epsilon)
    half = 0.5 * phi.sum(1, keepdim=True)
    margin = K[:, None] * ((half + phi).pow(power) - half.pow(power))
    if realistic:
        total = phi.sum(1, keepdim=True)
        at_empty = K[:, None] * phi.pow(power)
        at_full = K[:, None] * (total.pow(power) - (total - phi).clamp_min(0).pow(power))
        low, high = torch.minimum(at_empty, at_full).log(), torch.maximum(at_empty, at_full).log()
        u = 1.5 + torch.randn(n, c, generator=gen, dtype=DTYPE)
        cost = torch.exp(low + (high - low) * u)
    else:
        if near_symmetric:
            noise = torch.ones(n, c, dtype=DTYPE)
        else:
            noise = 0.5 + torch.rand(n, c, generator=gen, dtype=DTYPE)
        cost = cost_level * margin * noise
    return {"phi": phi, "scale": K, "cost": cost}, {"epsilon": torch.tensor(epsilon, dtype=DTYPE)}


# --------------------------------------------------------------------------
# Pairwise interactions
# --------------------------------------------------------------------------
def pairwise_profit(menus, data, params):
    linear = torch.einsum("bmc,bc->bm", menus, data["a"])
    return linear + 0.5 * torch.einsum("bmc,bcd,bmd->bm", menus, data["W"], menus)


def pairwise_marginal(sets, data, params):
    """D_j(S) = a_j + sum_{k in S, k != j} W_jk (zero diagonal), with a rounding bound."""
    D = data["a"][:, None, :] + torch.einsum("bjc,bkc->bkj", data["W"], sets)
    unit = 64 * torch.finfo(D.dtype).eps
    tol = unit * (data["a"].abs()[:, None, :] + torch.einsum("bjc,bkc->bkj", data["W"].abs(), sets))
    return D, tol


def make_pairwise(n, c, seed, sign, strength=1.0, density=1.0, integer=False):
    """A pairwise problem: sign=+1 complements (W >= 0), sign=-1 substitutes (W <= 0).

    Linear terms are centred so that roughly half the choices are worth taking
    once the interactions are counted. integer=True draws small integers, so
    many marginal values and menu values tie exactly.
    """
    gen = torch.Generator().manual_seed(seed)
    if integer:
        a = torch.randint(-2, 3, (n, c), generator=gen).to(DTYPE)
        upper = torch.randint(0, 2, (n, c, c), generator=gen).to(DTYPE)
    else:
        upper = strength * torch.randn(n, c, c, generator=gen, dtype=DTYPE).abs()
        upper = upper * (torch.rand(n, c, c, generator=gen, dtype=DTYPE) < density)
    W = torch.triu(upper, diagonal=1)
    W = sign * (W + W.transpose(1, 2))
    if not integer:
        a = torch.randn(n, c, generator=gen, dtype=DTYPE) - 0.5 * W.sum(2)
    return {"a": a, "W": W}, {}


# --------------------------------------------------------------------------
# Brute force and the pass rule
# --------------------------------------------------------------------------
def all_menus(c, device=None):
    """All 2**c menus as a [2**c, c] bool tensor."""
    return ((torch.arange(1 << c, device=device)[:, None] >> torch.arange(c, device=device)) & 1).bool()


def close(candidate, reference):
    """The profit tolerance: abs(candidate - reference) <= 1e-8 + 1e-8 * abs(reference)."""
    return (candidate - reference).abs() <= PROFIT_ATOL + PROFIT_RTOL * reference.abs()


def brute_force(profit, data, params, c, lower=None, upper=None):
    """Exact reference: evaluate every menu inside [lower, upper].

    Returns (value [N], menu [N, C], unique [N]). unique marks agents whose
    best menu is the only feasible menu within the profit tolerance of the
    best value; other menus, however poor, do not affect it.
    """
    patterns = all_menus(c)
    n = next(iter(data.values())).shape[0]
    values = profit(patterns[None].expand(n, -1, -1).to(DTYPE), data, params)
    feasible = torch.ones(values.shape, dtype=torch.bool)
    if lower is not None:
        feasible &= ~(lower[:, None, :] & ~patterns[None]).any(-1)
    if upper is not None:
        feasible &= ~(patterns[None] & ~upper[:, None, :]).any(-1)
    masked = torch.where(feasible, values, torch.full_like(values, -torch.inf))
    best, index = masked.max(1)
    assert torch.isfinite(best).all(), "reference optimum is not finite"
    unique = (feasible & close(values, best[:, None])).sum(1) == 1
    return best, patterns[index], unique


def profit_at(profit, menu, data, params, dtype=DTYPE):
    """Profit of one menu per agent ([N, C] bool -> [N])."""
    return profit(menu[:, None, :].to(dtype), data, params)[:, 0]


def check(sol, profit, data, params, value, menu=None, unique=None, lower=None, upper=None):
    """Assert the pass rule for a Solution against reference values (and menus)."""
    assert torch.isfinite(value).all(), "reference values are not finite"
    assert torch.isfinite(sol.value).all(), "solver returned non-finite values"
    bad = ~close(sol.value, value)
    assert not bad.any(), (f"{int(bad.sum())} agents miss the reference value; worst gap "
                           f"{float((sol.value - value).abs().max()):.3e}")
    if menu is not None:
        same = (sol.menu == menu).all(1) | ~unique
        assert same.all(), f"{int((~same).sum())} agents with a unique optimum got a different menu"
    reported = profit_at(profit, sol.menu, data, params, sol.value.dtype)
    assert torch.isfinite(reported).all(), "the reported menu's profit is not finite"
    assert close(sol.value, reported).all(), "reported value is not the menu's profit"
    if lower is not None:
        assert not (lower & ~sol.menu).any(), "menu drops a forced choice"
    if upper is not None:
        assert not (sol.menu & ~upper).any(), "menu includes a forbidden choice"
