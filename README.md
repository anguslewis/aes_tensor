# aes_tensor

A PyTorch implementation of the algorithm for solving combinatorial discrete
choice problems developed in Arkolakis, Eckert and Shi (forthcoming)
(hereafter AES), which builds on Jia (2008). In a combinatorial discrete
choice problem, each agent chooses the set of discrete choices (its menu)
that maximizes its value function (i.e. a profit function), where the value
of one choice depends on which other choices are made. This implementation
solves the problem for many heterogeneous agents at once, each with its own
characteristics for every choice. It is fully vectorized across agents, runs
natively on a GPU or a CPU, and takes the agents' profit function as ordinary
PyTorch code. This implementation of the algorithm was written for use in
Lewis and Xie (2026).

AES have also created a Julia package,
[CDCP.jl](https://github.com/rowanxshi/CDCP.jl), which may also be superior
for some applications. At the time of writing (CDCP.jl v1.0), it does not
support a vectorized, GPU-native solution for agents that are heterogeneous
across the discrete choices. As such, our vectorized implementation of the
algorithm will likely be computationally faster when there are large numbers
of agents with heterogeneity across choices and you have access to GPUs. In
other cases, the speed advantage may be small or the AES Julia package may be
faster. Relative performance will depend on both the application and
hardware.

If convenient for you, we request that if you use this package in your work
you cite Lewis and Xie (2026) as the source of the package.

## Installation

Install from GitHub by cloning this repository:

```bash
git clone https://github.com/anguslewis/aes_tensor.git
cd aes_tensor
```

**To run on GPUs**, create a conda environment with PyTorch installed;
PyTorch's installation instructions are at
<https://pytorch.org/get-started/locally/>. We assume this environment is
called `aes_tensor_gpu`. Then, from the repository root, install this package
and run the example with:

```bash
conda activate aes_tensor_gpu
pip install .
python examples/currency_choice.py
```

**For a quick install**, we also provide a conda environment:
`environment.yml`. We stress that, depending on your device, this
installation method may not install a version of PyTorch into the
environment that will utilize available GPUs. As such, we recommend
installing PyTorch into an environment yourself using the PyTorch
installation instructions above to ensure you are able to execute
`aes_tensor` using available GPUs. Nonetheless, this quick alternate
installation may be helpful, for example for local testing:

```bash
conda env create -f environment.yml
conda activate aes_tensor_quick
pip install .
python examples/currency_choice.py
```

## Quick start

```python
import torch
import aes_tensor

# Run on GPU if available, if not CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64

def profit(menus, data, params):
    """menus [B, M, C] of 0./1. -> profit [B, M]."""
    power = 1.0 / (1.0 + params["epsilon"])
    gross = data["K"][:, None] * torch.einsum("bmc,bc->bm", menus, data["phi"]).pow(power)
    return gross - torch.einsum("bmc,bc->bm", menus, data["cost"])

data = {"phi": phi.to(device, dtype),      # [N, C]
        "K": K.to(device, dtype),          # [N]
        "cost": cost.to(device, dtype)}    # [N, C]
params = {"epsilon": torch.tensor(0.5, dtype=dtype, device=device)}

sol = aes_tensor.solve(profit, data, params, n_choices=C, mode="substitutes", dtype=dtype)
sol.menu    # [N, C] bool: each agent's optimal menu
sol.value   # [N]: its profit
sol.stats   # work counters
```

`aes_tensor` implements the AES algorithm given a user-defined profit
function, which details the payoffs the agent receives for a given menu
choice. These payoffs can be determined by data (contained in the `data`
object) and parameters (contained in the `params` object). Data can vary
across agents, for example capturing preferences for certain choices. We
allow for either substitutability or complementarity between choices, as set
by `mode`.

We provide a more detailed example in `examples/currency_choice.py`, where we
solve the model in Lewis and Xie (2026) for a dataset of simulated firms. 

## Overview

The basic principle of the solution function is that you code up your
model's value function within a function called `profit` using only PyTorch
tensor operations. Tensor operations are similar to those of NumPy, and allow
for vectorized calculation of the value function for all agents. It is
crucial that `profit` is vectorized so that the solutions for heterogeneous
agents are computed in parallel on GPUs.

The value function must be a function of three arguments,
`profit(menus, data, params)`:

- `menus` holds the candidate menus to evaluate. It is a three-dimensional
  array of 0s and 1s. The first dimension indexes agents, the second indexes
  several candidate menus for that agent, and the third defines each menu
  across the C binary choices that make up a menu.
- `data` holds everything that differs across agents: for example, their
  productivity, or characteristics describing their preferences for certain
  choices. It is a dictionary of tensors, each indexed by agent along
  dimension 0, with any other dimensions your model needs.
- `params` holds the parameters common to all agents. These may be vectors,
  but they are not indexed by agent.

The `profit` function returns a two-dimensional array containing the value
function of each agent at each of its candidate menus (one row per agent, one
column per menu).

Once these objects are defined, you call the solver,
`aes_tensor.solve(profit, data, params, n_choices=..., mode=...)`. The solver
also needs the number of choices, `n_choices`, and `mode`, which states whether
the choices are complements or substitutes in your model. This sets the
direction in which the AES algorithm rules choices in and out, so it must
match a property of your profit function (see below). For each agent, `solve`
returns the optimal menu (`sol.menu`, one row of C true/false values per
agent), its profit (`sol.value`), and counts of the work done (`sol.stats`).

We also allow for an optional `marginal` function, which returns the change
in profit from adding or removing each choice directly. Without it, the solver
computes these marginal profits by differencing your profit function.
Depending on the functional form of the value function for your agents, a
formula for the marginal profits between menus may be computationally faster.

Finally, we provide a number of other options, including how far the solver
branches before taking the max over the remaining menus (`leaf_size`), the
starting lower and upper menus (`lower` and `upper`), and the floating-point
type and tolerance used (`dtype` and `rtol`).

The full solution function, written out with every argument, is given by:

```python
sol = aes_tensor.solve(
    profit, data, params,       # your profit function, per-agent data, common parameters
    n_choices=C,                # number of choices (required)
    mode="substitutes",         # or "complements" (required)
    lower=None,                 # [N, C] bool, choices forced in (default: none)
    upper=None,                 # [N, C] bool, choices allowed (default: all)
    marginal=None,              # optional marginal-profit function
    leaf_size=10,               # most undecided choices settled by trying every combination
    rtol=None,                  # tolerance (default: 64 x machine precision of dtype)
    dtype=torch.float64,        # floating-point type of menus and profits
)
```

`solve` returns a `Solution` with three fields, on the same device as your
data:

- `sol.menu`: an `[N, C]` true/false tensor, each agent's optimal menu
  (`True` = the choice is in the menu). If several menus are equally
  profitable, it is the first one found.
- `sol.value`: an `[N]` tensor, the profit of that menu, as computed by your
  profit function (in `dtype`).
- `sol.stats`: a dictionary recording the solver's work:
  - `squeeze_iterations`: total squeezing rounds across the initial solve and
    all branches;
  - `max_unresolved`: the most undecided choices any agent had after
    squeezing (if this is at most `leaf_size`, no agent branched);
  - `branched_rows`: how many times an agent's problem was split by the
    branching step;
  - `leaf_menus`: candidate menus whose profit was evaluated when settling
    agents by trying every remaining combination;
  - `profit_calls` and `marginal_calls`: calls to your `profit` and
    `marginal` functions.

We now provide additional details on each of the inputs and arguments of the
solution function.

## The profit function, data, and parameters

```python
profit(menus, data, params) -> Tensor [B, M]
```

- **`menus`** is a `[B, M, C]` tensor of 0./1. values (dtype `dtype`,
  `float64` by default): M candidate menus for each of B agents. B and M
  change from call to call, so the function must work for any B and M. Menus
  are stored as floats rather than booleans so that operations involving
  menus do not involve datatype switches, which may slow computation.
- **`data`** is your per-agent data restricted to the same B agents. `data`
  is a dictionary of tensors, each indexed by agent along dimension 0 (any
  further dimensions are yours). For example, `data["z"]`
  might be a `[B]` tensor containing productivities and `data["phi"]` a
  `[B, C]` tensor containing financing potentials that describe the
  preferences of firms across choices.
- **`params`** contains parameters that do not vary across agents. `params`
  can in principle be any object, since it is handed to the `profit` function
  untouched on every call within the solver. It is recommended you store all
  contents of `params` as tensors to ensure all computations are done on GPUs.
  For example, `params` might be a dictionary, with
  `params["epsilon"]` a scalar tensor containing the elasticity parameter
  epsilon and `params["kappa_c"]` a `[C]` tensor of the medians of the fixed
  cost distribution for each choice.

Rules:

1. **Row independence.** Row b of the output may depend only on row b of
   `data` and on `params`. The solver evaluates arbitrary subsets of agents.
2. **Menu independence.** Each of the M menus is evaluated on its own.
3. **Finite values** for every menu inside the agent's `[lower, upper]` bounds.
   The solver only ever evaluates menus with `lower <= menu <= upper`, so
   choices excluded by `upper` never need a sensible profit.

## Complements or substitutes: the `mode` argument

`mode` is required and applies to every agent in the call. It states the
direction of AES's single-crossing condition that your profit function
satisfies:

| `mode`                                  | AES's single-crossing condition                |
|-----------------------------------------|------------------------------------------------|
| `"complements"` (alias `"scdc_below"`)  | from below (implied by supermodularity)        |
| `"substitutes"` (alias `"scdc_above"`)  | from above (implied by submodularity)          |

We note that the solver does not allow for different agents to have different
modes, although the solver can be called on the subset of agents that
correspond to each mode separately.

We note also that the solver does not check the validity of the AES
conditions, and will attempt to solve the model even if the conditions do not
hold.

## Restrictions: `lower` and `upper`

Optional `[N, C]` bool tensors. `lower` marks choices forced into an agent's
menu (default: none); `upper` marks choices allowed at all (default: all).
`lower` must be a subset of `upper`.

For example, if some agents always include their "home" choice in their menu,
this can be set in `lower` so the algorithm never considers menus that omit
"home" for those agents. Likewise, if some agents are disallowed some choices,
this can be set in `upper` and the algorithm will never consider those choices
for those agents.

## Optional marginal-profit function: `marginal`

By default the solver gets marginal values by differencing your profit
function over single-choice toggles of each bound. Depending on the structure
of the value function in your model, it may be faster to compute the marginal
values directly. You can specify and pass a `marginal` function, which the
algorithm will use for every step up to the final maximization across
remaining menus:

```python
marginal(sets, data, params) -> D  or  (D, tol)
```

- **`sets`** is a `[B, M, C]` tensor of 0./1. values: for each of B agents,
  the M menus at which the solver needs marginal values. These are the bounds
  of AES's squeezing step. Currently M = 2: the agent's lower bound (the
  choices already known to be in its optimal menu) and its upper bound (the
  choices that may still be in it). Write the function to work for any M.
- **`data`** and **`params`** are as for `profit`.
- **`D`** is a `[B, M, C]` tensor of marginal values: `D[b, m, c]` is agent
  b's profit from menu `sets[b, m]` with choice c added, minus its profit from
  that menu with choice c removed, whether or not c is in `sets[b, m]`.
- **`tol`** (optional) is a non-negative `[B, M, C]` tensor bounding the
  rounding error in `D`: the solver treats a marginal value no larger than
  `tol` in absolute value as zero. Without it, comparisons are exact.

Only the entries for choices that are still undecided are used. The profit
function is still required: the final maximization across remaining menus,
and the values reported in `sol.value`, use profit levels. The `marginal`
function in `examples/currency_choice.py` shows one way to write it.

## Branching: `leaf_size`

`leaf_size` controls at what point the solution algorithm defaults to a simple
brute-force maximization over the remaining menus:

1. The solver first applies AES's squeezing step to every agent. This step is
   vectorized across agents and can benefit from GPU acceleration.
2. For agents left with more than `leaf_size` undecided choices, it then
   applies AES's branching step. This step iteratively splits the agent's
   problem into two branches on its first undecided choice in column order
   (with that choice ruled out in one branch and ruled in in the other), then
   squeezes each branch. This iterative process continues until every branch
   has at most `leaf_size` undecided choices. Each individual squeeze step is
   fully vectorized on GPUs, but across iterations there are many GPU calls in
   sequence, many of which might not fully occupy the GPU. As a consequence,
   this step can lose some computational efficiency.
3. The remaining undecided choices are settled by evaluating the profit of
   every remaining combination of choices and taking the maximum. This
   brute-force step also evaluates agents and candidate menus in parallel on
   GPUs. However, it evaluates up to `2**leaf_size` menus for each
   agent at once, so both its run time and its memory use double with each
   additional undecided choice.

We include this option because there are potential trade-offs in how much
branching is done. A smaller `leaf_size` reduces the memory and computation
requirements of Step 3, at the cost of shifting more of the solution work into
the potentially less computationally efficient Step 2. The best value of
`leaf_size` may therefore vary widely with the problem being solved, because
the relative strength of these forces depends on the specific model and
hardware: for
example, on how many choices remain undecided after squeezing and how much
each round of branching settles.

We set the default at `leaf_size=10` (at most 1,024 combinations per agent).
`leaf_size=0` branches all the way down, as in AES, and `leaf_size=n_choices`
never branches. For large numbers of agents facing many choices, memory
constraints may require a lower `leaf_size`. For relatively small numbers of
agents facing relatively few choices, `leaf_size=n_choices` may be the optimal
setting for computational speed.

## Data types and tolerance: `dtype` and `rtol`

`dtype` is the floating-point type of the menus passed to your profit
function and of the profits it returns. `dtype` must be `torch.float32` or
`torch.float64`; the default is `torch.float64`. Compared with
`torch.float64`, `torch.float32` halves floating tensor element size and may
run faster, at the cost of precision. Some GPUs do not support `float64`.

`rtol` guards the squeezing step against rounding error. When the solver
computes a choice's marginal value by differencing your profit function, it
leaves the choice undecided whenever that marginal value is smaller than
`rtol` times the size of the two profits being differenced. This does not
change the solution: it only determines whether a choice is settled during
squeezing or left to be settled later by branching or brute force. The default
is 64 times the machine precision of `dtype`, and `rtol=0` gives exact
comparisons. In `float64` the tolerance rarely matters; it matters more in
`float32`, where rounding errors are many orders of magnitude larger.

`rtol` applies only when the solver differences your profit function. If you
supply a `marginal` function, the solver does not see the profits behind its
marginal values, so your `marginal` can return its own bound, `tol`, instead
(see above); without one, comparisons are exact.

## Errors

`solve` raises on malformed inputs or outputs: `data` that is not a
dictionary of tensors, or whose tensors disagree on the number of agents or
device; an unknown `mode` or invalid option, including a `dtype` other than
`torch.float32` or `torch.float64`; bounds of the wrong shape, type or
device, or with `lower` not inside `upper`; and profit or marginal output of
the wrong shape, dtype or device. It also raises on non-finite profit or
marginal values (or a negative marginal `tol`) for the menus and choices the
solver uses, and on a non-finite marginal value or tolerance computed during
squeezing, for example when profits are close to the largest number `dtype`
can represent; if this happens, rescale your profit function or inputs.
Finally, as a safeguard, it raises if squeezing fails to converge within
`2C + 1` iterations. Nothing else is checked or caught.

## Memory

The algorithm can be memory hungry for larger samples, and you may need to
solve in batches of agents; `aes_tensor` does no batching of its own.
Lowering `leaf_size` reduces the memory used when trying every remaining
combination, at the potential cost of more branching; it does not reduce the
memory used by the squeezing step.

## Tests

```bash
pytest            # from the repository root, in an environment with pytest installed
```

The test suite checks the solver against brute-force enumeration of every menu
(up to 12 choices) for CES-aggregator and pairwise-interaction objectives in
both modes, across leaf sizes, restrictions, ties, near-ties, extreme
parameter values, profit scales and `float32`; checks the profit-only and
marginal-function paths against each other at 27 and 40 choices; and covers
the interface, input validation, numerical errors and determinism. Against
brute force, the returned menu must match exactly wherever the optimum is
unique, and profits must agree to within an absolute and a relative tolerance
of 1e-8. Tests comparing CPU and GPU results run only where CUDA is
available. Developed and tested with Python 3.11 and PyTorch 2.6.0.

## References

- Arkolakis, C., F. Eckert and R. Shi (forthcoming). "Combinatorial Discrete
  Choice: Theory and Application to Multinational Production." *American
  Economic Review*.
- Jia, P. (2008). "What Happens When Wal-Mart Comes to Town: An Empirical
  Analysis of the Discount Retailing Industry." *Econometrica* 76(6).
- Lewis, A. and O. Xie (2026). "Firm-Level Gains from Financial Integration."
  Working paper.

## License

`aes_tensor` is released under the MIT License; see [`LICENSE`](LICENSE).
