"""Branching and leaf enumeration around the squeezing step.

After squeezing, each agent has some number of undecided choices
(``upper & ~lower``). Agents with at most ``leaf_size`` undecided choices are
settled by evaluating every completion of their lower bound (a "leaf").
Agents with more are split by the AES branching step on their first undecided
choice in column order: one child with the choice ruled out, one with it
ruled in. Each child is squeezed again, conditional on that choice.

The tree is traversed depth first with an explicit stack. Every node holds
only the agents still unresolved along its path, as ``rows`` (indices into
the full problem) plus their [n, C] bounds, so no tensor carries a tree
dimension. Each agent's best (menu, value) over all its leaves is kept.
"""
import torch

from . import rows as row_ops
from .marginals import evaluate_profit
from .squeeze import squeeze


def solve_tree(profit, marginal_values, data, params, lower, upper, substitutes,
               leaf_size, dtype, stats):
    """Return the optimal menu [N, C] bool and its value [N] for every agent.

    marginal_values(node_data, sets, free) -> (D, tol) supplies the marginal
    values for the agents at a node.
    """
    n, c = lower.shape
    device = lower.device
    best_menu = torch.zeros_like(lower)
    best_value = torch.full((n,), -torch.inf, dtype=dtype, device=device)
    stack = [(torch.arange(n, device=device), lower, upper)]
    while stack:
        rows, lo, hi = stack.pop()
        node_data = row_ops.take(data, rows)
        lo, hi, iterations = squeeze(
            lambda sets, free: marginal_values(node_data, sets, free), lo, hi, substitutes)
        stats["squeeze_iterations"] += iterations
        free_count = (hi & ~lo).sum(1)
        max_free = int(free_count.max())
        stats["max_unresolved"] = max(stats["max_unresolved"], max_free)
        # Leaves, grouped by number of undecided choices so each group's
        # candidate menus form one [b, 2**size, C] tensor.
        for size in range(min(leaf_size, max_free) + 1):
            local = torch.nonzero(free_count == size).squeeze(1)
            if local.numel() == 0:
                continue
            menu, value = enumerate_leaf(profit, row_ops.take(node_data, local), params,
                                         lo[local], hi[local], size, dtype, stats)
            target = rows[local]
            improve = value > best_value[target]
            best_value[target] = torch.where(improve, value, best_value[target])
            best_menu[target] = torch.where(improve[:, None], menu, best_menu[target])
        # Branch the rest on their first undecided choice in column order.
        unresolved = torch.nonzero(free_count > leaf_size).squeeze(1)
        if unresolved.numel():
            stats["branched_rows"] += int(unresolved.numel())
            ulo, uhi = lo[unresolved], hi[unresolved]
            choice = (uhi & ~ulo).long().argmax(1)[:, None]
            ruled_out = uhi.clone()
            ruled_out.scatter_(1, choice, False)
            ruled_in = ulo.clone()
            ruled_in.scatter_(1, choice, True)
            # LIFO: the ruled-out child is processed first, then the ruled-in child.
            stack.append((rows[unresolved], ruled_in, uhi))
            stack.append((rows[unresolved], ulo, ruled_out))
    return best_menu, best_value


def enumerate_leaf(profit, data, params, lower, upper, size, dtype, stats):
    """Best completion for agents that each have exactly ``size`` undecided choices.

    The 2**size on/off patterns are defined over the agents' undecided slots,
    not over choices, so one pattern matrix serves every agent even though
    their undecided choices differ. Returns the best menu [b, C] bool and its
    value [b]; ties go to the first pattern in enumeration order.
    """
    b, c = lower.shape
    count = 1 << size
    slots = torch.nonzero(upper & ~lower)[:, 1].reshape(b, size)          # [b, size]
    patterns = ((torch.arange(count, device=lower.device)[:, None]
                 >> torch.arange(size, device=lower.device)) & 1).bool()   # [count, size]
    menus = lower[:, None, :].expand(b, count, c).clone()
    menus.scatter_(2, slots[:, None, :].expand(b, count, size), patterns[None].expand(b, count, size))
    values = evaluate_profit(profit, menus.to(dtype), data, params, stats)
    stats["leaf_menus"] += b * count
    value, choice = values.max(1)
    return menus[torch.arange(b, device=lower.device), choice], value
