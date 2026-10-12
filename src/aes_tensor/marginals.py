"""Profit evaluation and marginal values for the squeezing step.

The squeezing step needs, for every agent b, bound menu m and choice c, the
marginal value

    D[b, m, c] = pi_b(menu m with c) - pi_b(menu m without c)

together with a numerical tolerance ``tol``: a marginal value with
``|D| <= tol`` is treated as undecided, a practical guard against rounding
error ruling a choice in or out. Two sources are supported:

* ``from_profit`` (generic): differences the user's profit function over
  single-choice toggles of each bound set.
* ``from_marginal``: calls a user-supplied marginal-profit function that
  exploits the structure of the objective.

The squeezing step checks that the entries it uses are finite.
"""
import torch


def evaluate_profit(profit, menus, data, params, stats):
    """Call ``profit`` on ``menus`` [B, M, C] and validate its [B, M] output."""
    stats["profit_calls"] += 1
    values = profit(menus, data, params)
    expected = tuple(menus.shape[:2])
    if not isinstance(values, torch.Tensor) or tuple(values.shape) != expected:
        got = tuple(values.shape) if isinstance(values, torch.Tensor) else type(values).__name__
        raise ValueError(f"profit must return a tensor of shape {expected} (agents x menus); got {got}")
    if values.dtype != menus.dtype or values.device != menus.device:
        raise TypeError(f"profit must return {menus.dtype} values on {menus.device}; "
                        f"got {values.dtype} on {values.device}")
    if not torch.isfinite(values).all():
        raise FloatingPointError("profit returned non-finite values for menus inside the "
                                 "[lower, upper] bounds")
    return values


def from_profit(profit, data, params, sets, free, rtol, dtype, stats):
    """Marginal values by differencing profit levels (one profit call).

    sets: [B, M, C] bool bound menus; free: [B, C] bool undecided choices.
    For each bound menu, the menus evaluated are that menu itself plus the
    menu with choice c toggled, for every undecided c. Decided choices are "toggled" to a no-op,
    so profit is only ever evaluated on menus inside the current bounds.
    Returns D and tol, each [B, M, C]; entries for decided choices are unused.
    """
    b, m, c = sets.shape
    eye = torch.eye(c, dtype=torch.bool, device=sets.device)
    flips = eye[None, :, :] & free[:, :, None]                            # [B, C (toggled), C]
    toggled = sets[:, :, None, :] ^ flips[:, None, :, :]                 # [B, M, C (toggled), C]
    menus = torch.cat((sets[:, :, None, :], toggled), dim=2)              # [B, M, 1+C, C]
    values = evaluate_profit(profit, menus.reshape(b, m * (c + 1), c).to(dtype), data, params, stats)
    values = values.reshape(b, m, c + 1)
    base, other = values[..., :1], values[..., 1:]                        # [B, M, 1], [B, M, C]
    # c in the menu: other = pi(menu without c); c not in it: other = pi(menu with c).
    D = torch.where(sets, base - other, other - base)
    tol = rtol * (base.abs() + other.abs())
    return D, tol


def from_marginal(marginal, data, params, sets, free, dtype, stats):
    """Marginal values from the user's ``marginal(sets, data, params)``.

    The user function receives the bound menus as [B, M, C] 0./1. values of the
    solver dtype and returns either D or (D, tol), each [B, M, C]. Without a
    tolerance, comparisons are exact. Only entries for undecided choices are
    used, so only those must be finite.
    """
    stats["marginal_calls"] += 1
    out = marginal(sets.to(dtype), data, params)
    if isinstance(out, tuple):
        if len(out) != 2:
            raise ValueError(f"marginal must return D or (D, tol); got a tuple of length {len(out)}")
        D, tol = out
    else:
        D, tol = out, None
    expected = tuple(sets.shape)
    for name, value in (("D", D), ("tol", tol)):
        if name == "tol" and value is None:
            continue
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected:
            got = tuple(value.shape) if isinstance(value, torch.Tensor) else type(value).__name__
            raise ValueError(f"marginal {name} must have shape {expected} (agents x menus x choices); got {got}")
        if value.dtype != dtype or value.device != sets.device:
            raise TypeError(f"marginal {name} must be {dtype} on {sets.device}; "
                            f"got {value.dtype} on {value.device}")
    used = free[:, None, :].expand(expected)
    if not torch.isfinite(D[used]).all():
        raise FloatingPointError("marginal returned non-finite D for undecided choices")
    if tol is None:
        tol = torch.zeros_like(D)
    elif not torch.isfinite(tol[used]).all() or (tol[used] < 0).any():
        raise FloatingPointError("marginal tol must be finite and non-negative for undecided choices")
    return D, tol
