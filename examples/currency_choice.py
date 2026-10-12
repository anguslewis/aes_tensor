"""Example: firms choosing which currencies to borrow in (Lewis and Xie 2026).

Each firm chooses the set of currencies it borrows in (its "menu") to
maximize profit. With epsilon > 0 the currencies are substitutes; with
-1 < epsilon < 0 they are complements.

This file has three parts:

    1. The model-implied functions, profit and marginal. These are the
       inputs to the solver.
    2. simulate_firms, which makes up firm data for this example.
    3. main, which solves the model from start to finish, twice: first the
       simplest case, then with the home currency always entered and
       capital controls.

Each problem is solved twice, once giving the solver only the profit
function and once also giving it the marginal-profit function. The two must
give identical answers. Depending on the functional form of the value
function, supplying marginal profits may speed up the solution.

    python examples/currency_choice.py
"""
import torch

import aes_tensor

CURRENCIES = ["USD", "EUR", "JPY", "GBP", "CHF", "CAD", "AUD", "SEK", "NOK", "NZD", "TWD", "INR"]


# ==========================================================================
# 1. Model-implied functions. These are the inputs to the solver.
# ==========================================================================

def profit(menus, data, params):
    """Calculates firm profits as in Lewis and Xie (2026).

    For firm j and a menu M (a set of currencies), profit net of fixed costs is

        profit_j(M) = K_j * (sum of Phi_jc over the currencies c in M) ** (1 / (1 + epsilon))
                      - (sum of kappa_jc over the currencies c in M),

    where K_j is a function of the firm's productivity and parameter values,
    Phi_jc is firm j's financing potential in currency c, and kappa_jc is the
    fixed cost firm j must pay to enter currency c.

    Inputs:
        menus:  the menus to evaluate, an array of 0s and 1s with one row per
                firm, one column per menu, and one entry per currency
                (1 = the currency is in that menu).
        data:   the firms' data: K (K_j), phi (Phi_jc), kappa (kappa_jc).
        params: the parameter values: epsilon.
    Output:
        the profit of every menu for every firm (one row per firm, one
        column per menu).
    """
    phi_sum = torch.einsum("jmc,jc->jm", menus, data["phi"])
    total_fixed_cost = torch.einsum("jmc,jc->jm", menus, data["kappa"])
    return data["K"][:, None] * phi_sum ** (1 / (1 + params["epsilon"])) - total_fixed_cost


def marginal(sets, data, params):
    """Calculates the marginal profit of each currency as in Lewis and Xie (2026).

    For firm j, a menu M and a currency c, the marginal profit is the change
    in profit from having c in the menu versus not having it:

        D_jc(M) = profit_j(M with c) - profit_j(M without c)
                = K_j * ((A + Phi_jc) ** (1 / (1 + epsilon)) - A ** (1 / (1 + epsilon))) - kappa_jc,

    where A is the sum of Phi over the other currencies in M, and K_j, Phi_jc
    and kappa_jc are as in profit.

    Inputs:
        sets:   the menus at which to evaluate marginal profit, an array of 0s
                and 1s with one row per firm, one column per menu, and one
                entry per currency.
        data:   the firms' data: K (K_j), phi (Phi_jc), kappa (kappa_jc).
        params: the parameter values: epsilon.
    Outputs:
        D:   the marginal profit of every currency, at every menu, for every
             firm (same shape as sets).
        tol: a bound on the rounding error in D; the solver treats a
             marginal profit smaller than tol as zero.
    """
    power = 1 / (1 + params["epsilon"])
    phi = data["phi"][:, None, :]
    kappa = data["kappa"][:, None, :]
    in_menu = sets * phi
    zero = torch.zeros_like(in_menu[..., :1])
    before = torch.cat((zero, in_menu[..., :-1].cumsum(-1)), -1)
    after = torch.cat((in_menu[..., 1:].flip(-1).cumsum(-1).flip(-1), zero), -1)
    A = before + after
    K = data["K"][:, None, None]
    D = K * ((A + phi) ** power - A ** power) - kappa
    unit = 64 * torch.finfo(D.dtype).eps
    tol = unit * K * ((A + phi) ** power + A ** power) + unit * kappa
    return D, tol


# ==========================================================================
# 2. Simulated firm data for this example.
# ==========================================================================

def simulate_firms(n_firms, n_home, seed, params, device):
    """Simulates data for n_firms firms, one row per firm.

    Returns data, with
        K:       K_j (a function of productivity and parameter values, drawn
                 as if it were a primitive for this example).
        phi:     the firm's financing potential Phi_jc in each currency.
        kappa:   the fixed cost kappa_jc of entering each currency, drawn as in
                 the model: lognormal with a currency-specific median kappa_c
                 and a standard deviation sigma of log fixed costs common to
                 all currencies. The medians are set higher when currencies
                 are complements (epsilon < 0), so that firms do not simply
                 enter every currency.
        is_home: marks the firm's home currency, one of the first n_home
                 currencies.
    """
    n_currencies = len(CURRENCIES)
    gen = torch.Generator().manual_seed(seed)
    home = torch.randint(0, n_home, (n_firms,), generator=gen)
    K = 10.0 * torch.exp(0.5 * torch.randn(n_firms, generator=gen, dtype=torch.float64))
    phi = torch.exp(torch.randn(n_firms, n_currencies, generator=gen, dtype=torch.float64))
    u = torch.randn(n_firms, n_currencies, generator=gen, dtype=torch.float64)
    kappa_low, kappa_high = (1.0, 4.0) if float(params["epsilon"]) > 0 else (20.0, 80.0)
    kappa_c = torch.linspace(kappa_low, kappa_high, n_currencies, dtype=torch.float64)
    sigma = 0.5
    data = {
        "K": K,
        "phi": phi,
        "kappa": kappa_c * torch.exp(sigma * u),
        "is_home": torch.nn.functional.one_hot(home, n_currencies).bool(),
    }
    return {name: value.to(device) for name, value in data.items()}


# ==========================================================================
# 3. Solving the model, from start to finish.
# ==========================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    n_currencies = len(CURRENCIES)

    # First we do the simplest example. Every currency is open to every firm,
    # and every currency, including the firm's home currency, has a fixed
    # cost. Nothing is forced in or ruled out in advance, so lower and upper
    # are left at their defaults: lower is all 0s (no currency forced into the
    # menu) and upper is all 1s (every currency allowed).
    print("Example 1: every currency open, every currency costly")
    for mode, epsilon in (("substitutes", 1.0), ("complements", -0.3)):
        params = {"epsilon": torch.tensor(epsilon, dtype=torch.float64, device=device)}
        data = simulate_firms(n_firms=2_000, n_home=4, seed=0, params=params, device=device)
        sol_profit = aes_tensor.solve(profit, data, params, n_choices=n_currencies, mode=mode)
        sol_marginal = aes_tensor.solve(profit, data, params, n_choices=n_currencies, mode=mode,
                                        marginal=marginal)
        assert torch.equal(sol_profit.menu, sol_marginal.menu) and torch.allclose(sol_profit.value, sol_marginal.value, rtol=1e-12)
        print(f"\n  {mode} (epsilon = {epsilon})")
        print("  Profit stats:  ", sol_profit.stats)
        print("  Marginal stats:", sol_marginal.stats)
        print("  Mean currencies entered:", round(float(sol_profit.menu.sum(1).double().mean()), 2))
        print("  First firm's optimal menu:", [c for c, entered in zip(CURRENCIES, sol_profit.menu[0]) if entered])
        print("  First firm's optimal profit:", round(float(sol_profit.value[0]), 4))

    # Now we do a slightly more complicated case. Every firm always borrows in
    # its home currency, and some currencies are closed to entry (capital
    # controls): here TWD and INR, which are never a firm's home currency in
    # this example. This involves modifying the lower/upper starting points.
    # The home currency is set in lower, so it is always included; the
    # capital-control currencies are removed from upper, so they are never
    # included. Neither is ever up for decision, and the profit function is
    # never evaluated on a menu that contains a closed currency.
    print("\nExample 2: home currency always entered, capital controls on TWD and INR")
    for mode, epsilon in (("substitutes", 1.0), ("complements", -0.3)):
        params = {"epsilon": torch.tensor(epsilon, dtype=torch.float64, device=device)}
        data = simulate_firms(n_firms=2_000, n_home=4, seed=0, params=params, device=device)
        lower = data["is_home"]
        upper = torch.ones_like(lower)
        upper[:, CURRENCIES.index("TWD")] = False
        upper[:, CURRENCIES.index("INR")] = False
        sol_profit = aes_tensor.solve(profit, data, params, n_choices=n_currencies, mode=mode,
                                      lower=lower, upper=upper)
        sol_marginal = aes_tensor.solve(profit, data, params, n_choices=n_currencies, mode=mode,
                                        lower=lower, upper=upper, marginal=marginal)
        assert torch.equal(sol_profit.menu, sol_marginal.menu) and torch.allclose(sol_profit.value, sol_marginal.value, rtol=1e-12)
        print(f"\n  {mode} (epsilon = {epsilon})")
        print("  Profit stats:  ", sol_profit.stats)
        print("  Marginal stats:", sol_marginal.stats)
        print("  Mean currencies entered:", round(float(sol_profit.menu.sum(1).double().mean()), 2))
        print("  First firm's optimal menu:", [c for c, entered in zip(CURRENCIES, sol_profit.menu[0]) if entered])
        print("  First firm's optimal profit:", round(float(sol_profit.value[0]), 4))


if __name__ == "__main__":
    main()
