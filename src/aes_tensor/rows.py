"""Row selection for the per-agent ``data`` dictionary.

``data`` is a dictionary of tensors, each indexed by agent along dimension 0.
The solver repeatedly restricts the problem to a subset of agents (the agents
with a given number of undecided choices, the agents still unresolved at a
branch node), so it takes the same rows of every tensor.

Only dictionaries are accepted. Rebuilding other containers with selected rows
can run user code (a dataclass's ``__post_init__``, for example) that changes
the stored values, so they are rejected rather than reconstructed.
"""
import torch


def check(data):
    """Validate ``data`` and return its number of agents and device.

    ``data`` must be a dictionary whose every value is a tensor of at least one
    dimension, all with the same first dimension (the number of agents) and on
    the same device. Values shared by all agents belong in ``params``.
    """
    if not isinstance(data, dict):
        raise TypeError("data must be a dictionary of tensors, each indexed by agent along "
                        f"dimension 0; got {type(data).__name__}. Convert other containers "
                        "(for example a dataclass) to a dictionary first")
    if not data:
        raise ValueError("data must contain at least one tensor with agents along dimension 0")
    first_name, n, device = None, None, None
    for name, value in data.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"data field {name!r} is a {type(value).__name__}, not a tensor; "
                            "values shared by all agents belong in params")
        if value.ndim == 0:
            raise ValueError(f"data field {name!r} is a 0-d tensor; every data tensor needs agents "
                             "along dimension 0 (values shared by all agents belong in params)")
        if n is None:
            first_name, n, device = name, value.shape[0], value.device
        elif value.shape[0] != n:
            raise ValueError(f"data field {name!r} has {value.shape[0]} rows along dimension 0 but "
                             f"{first_name!r} has {n}; every data tensor needs one row per agent")
        elif value.device != device:
            raise ValueError(f"data field {name!r} is on {value.device} but {first_name!r} is on {device}")
    return n, device


def take(data, rows):
    """Return a new dictionary holding ``value[rows]`` for every tensor in ``data``."""
    return {name: value[rows] for name, value in data.items()}
