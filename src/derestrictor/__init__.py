"""Derestrictor: norm-preserving orthogonal-projection derestriction toolkit.

Removes refusal behavior from transformer language models via null-space
constrained, adaptively weighted orthogonal projection onto the refusal
direction. See the project README for the mathematical background and
references.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("derestrictor")
except PackageNotFoundError:
    __version__ = "0.0.0+local"

__all__ = ["__version__"]
