"""Public entry point: ``solve`` and ``Solution``."""
import math
from dataclasses import dataclass

import torch

from . import marginals
from . import rows as row_ops
from .branch import solve_tree

# mode name -> True for substitutes (SCD-C from above), False for complements (SCD-C from below)
MODES = {
    "complements": False,
    "scdc_below": False,
    "substitutes": True,
    "scdc_above": True,
}


@dataclass(frozen=True)
class Solution:
    """Result of ``solve``.

    menu:  [N, C] bool, the optimal menu of each agent.
    value: [N], the profit of that menu.
    stats: work counters (squeeze_iterations, max_unresolved, branched_rows,
           leaf_menus, profit_calls, marginal_calls).
    """
    menu: torch.Tensor
    value: torch.Tensor
    stats: dict


def _check_bound(name, bound, n, n_choices, device):
    if not isinstance(bound, torch.Tensor) or bound.dtype != torch.bool:
        raise TypeError(f"{name} must be a bool tensor")
    if tuple(bound.shape) != (n, n_choices):
        raise ValueError(f"{name} must have shape {(n, n_choices)} (agents x choices); got {tuple(bound.shape)}")
    if bound.device != device:
        raise ValueError(f"{name} is on {bound.device} but data is on {device}")


@torch.no_grad()
def solve(profit, data, params, *, n_choices, mode, lower=None, upper=None, marginal=None,
          leaf_size=10, rtol=None, dtype=torch.float64):
    """Solve every agent's combinatorial discrete choice problem.

    Each agent b picks the menu S (a subset of the n_choices choices) that
    maximizes profit, using the AES squeezing step, AES branching on the
    first undecided choice in column order, and enumeration once at most
    ``leaf_size`` choices remain undecided.

    The result is exact (up to floating point) only if every agent's
    objective satisfies the single-crossing condition for ``mode`` on its
    domain [lower, upper]. That is the caller's responsibility: nothing here
    checks it, and there is no brute-force fallback.

    Args:
        profit: ``profit(menus, data, params) -> [B, M]``. ``menus`` is a
            [B, M, C] tensor of 0./1. (dtype ``dtype``) holding M candidate
            menus for each of B agents; ``data`` holds the same B agents' rows
            (B and M vary between calls); ``params`` is passed through as
            given. Row b of the output may depend only on row b of ``data``
            and on ``params``.
        data: a dictionary of tensors, each indexed by agent along dimension
            0. Other containers (dataclasses, NamedTuples, other mappings)
            are rejected.
        params: anything (intended: tensors); passed to ``profit`` and
            ``marginal`` untouched and never sliced.
        n_choices: number of choices C.
        mode: "complements" (SCD-C from below; alias "scdc_below") or
            "substitutes" (SCD-C from above; alias "scdc_above"). Required.
        lower: optional [N, C] bool; choices forced into every menu.
            Default: none.
        upper: optional [N, C] bool; choices allowed at all. Default: all.
            Profit is only ever evaluated on menus with lower <= menu <= upper.
        marginal: optional marginal-profit function
            ``marginal(sets, data, params) -> D or (D, tol)`` for the
            squeezing step. ``sets`` is [B, M, C] of
            0./1.; D[b, m, c] = pi_b(sets[b, m] with c) - pi_b(sets[b, m]
            without c). Without tol, comparisons are exact. If omitted,
            marginal values are computed by differencing ``profit``.
        leaf_size: maximum number of undecided choices settled by enumerating
            all 2**k completions (0 = pure AES branching).
        rtol: relative tolerance for marginal values derived from ``profit``;
            |D| <= rtol * (|pi(menu with c)| + |pi(menu without c)|) leaves the
            choice undecided. Default 64 * machine epsilon of ``dtype``;
            0 gives exact comparisons.
        dtype: torch.float32 or torch.float64, the type of menus and
            profits. Default torch.float64.

    Returns:
        Solution(menu [N, C] bool, value [N], stats dict).
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {sorted(MODES)}; got {mode!r}")
    if not isinstance(n_choices, int) or isinstance(n_choices, bool) or n_choices < 1:
        raise ValueError(f"n_choices must be a positive int; got {n_choices!r}")
    if not isinstance(leaf_size, int) or isinstance(leaf_size, bool) or leaf_size < 0:
        raise ValueError(f"leaf_size must be a non-negative int; got {leaf_size!r}")
    if dtype not in (torch.float32, torch.float64):
        raise TypeError(f"dtype must be torch.float32 or torch.float64; got {dtype!r}")
    if rtol is None:
        rtol = 64 * torch.finfo(dtype).eps
    if not isinstance(rtol, (int, float)) or not math.isfinite(rtol) or rtol < 0:
        raise ValueError(f"rtol must be a finite non-negative number; got {rtol!r}")
    n, device = row_ops.check(data)
    if lower is None:
        lower = torch.zeros((n, n_choices), dtype=torch.bool, device=device)
    _check_bound("lower", lower, n, n_choices, device)
    if upper is None:
        upper = torch.ones((n, n_choices), dtype=torch.bool, device=device)
    _check_bound("upper", upper, n, n_choices, device)
    if (lower & ~upper).any():
        raise ValueError("lower must be a subset of upper for every agent")

    stats = {"squeeze_iterations": 0, "max_unresolved": 0, "branched_rows": 0,
             "leaf_menus": 0, "profit_calls": 0, "marginal_calls": 0}
    if n == 0:
        return Solution(lower.clone(), torch.empty(0, dtype=dtype, device=device), stats)
    if marginal is None:
        def marginal_values(node_data, sets, free):
            return marginals.from_profit(profit, node_data, params, sets, free, rtol, dtype, stats)
    else:
        def marginal_values(node_data, sets, free):
            return marginals.from_marginal(marginal, node_data, params, sets, free, dtype, stats)
    menu, value = solve_tree(profit, marginal_values, data, params, lower, upper,
                             MODES[mode], leaf_size, dtype, stats)
    return Solution(menu, value, stats)
