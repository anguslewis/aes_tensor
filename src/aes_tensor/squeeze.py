"""The AES squeezing step, vectorized over agents.

Each agent carries a pair of bounds ``lower`` (choices known to be in the
optimal menu) and ``upper`` (choices that may still be in it), both [B, C]
bool. One iteration evaluates every choice's marginal value at both bounds
and applies AES's squeezing map; which bound decides inclusion
and which decides exclusion depends on the mode:

    mode          rule in choice c if        rule out choice c if
    complements   D_c(lower) >  tol          D_c(upper) < -tol
    substitutes   D_c(upper) >  tol          D_c(lower) < -tol

Marginal values within the tolerance band (including exact zeros) leave the
choice undecided; it is then settled by branching or enumeration. Bounds only
ever tighten (lower grows, upper shrinks), so iteration stops at a fixed point.

Before any decision, the marginal values and tolerances of undecided choices
must be finite (finite profits can still overflow when differenced or summed);
otherwise the solver stops rather than rule a choice in or out on a NaN or an
infinity.
"""
import torch

NON_FINITE = ("Non-finite marginal value or tolerance encountered. "
              "Rescale your profit function or inputs.")


def squeeze(marginal_values, lower, upper, substitutes):
    """Iterate the squeezing map to its fixed point.

    marginal_values(sets, free) -> (D, tol): sets is [B, 2, C] with the lower
    bound at index 0 and the upper bound at index 1; free = upper & ~lower.
    Returns the tightened (lower, upper) and the number of iterations run.
    """
    c = lower.shape[1]
    for iteration in range(1, 2 * c + 2):
        free = upper & ~lower
        if not free.any():
            return lower, upper, iteration - 1
        D, tol = marginal_values(torch.stack((lower, upper), 1), free)
        # Only undecided choices' entries, at both bounds, drive decisions.
        used = free[:, None, :]
        if not ((torch.isfinite(D) & torch.isfinite(tol)) | ~used).all():
            raise FloatingPointError(NON_FINITE)
        if substitutes:
            rule_in = D[:, 1] > tol[:, 1]
            keep = D[:, 0] >= -tol[:, 0]
        else:
            rule_in = D[:, 0] > tol[:, 0]
            keep = D[:, 1] >= -tol[:, 1]
        new_lower = lower | (rule_in & upper)
        new_upper = upper & (keep | lower)
        if torch.equal(new_lower, lower) and torch.equal(new_upper, upper):
            return lower, upper, iteration
        lower, upper = new_lower, new_upper
    raise RuntimeError(f"AES squeezing did not converge within {2 * c + 1} iterations")
