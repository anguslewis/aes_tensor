"""aes_tensor: batched PyTorch solver for combinatorial discrete choice problems.

Implements the squeezing and branching procedure of Arkolakis, Eckert & Shi
("Combinatorial Discrete Choice", NBER WP 31877) for many agents at once.
See README.md for the profit-function contract and the validity assumption.
"""
from .api import MODES, Solution, solve

__all__ = ["MODES", "Solution", "solve"]
__version__ = "0.1.0"
