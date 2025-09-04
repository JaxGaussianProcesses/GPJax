"""Linear algebra module for GPJax.

This module provides lightweight LinearOperator abstractions and operations
to replace the cola dependency.
"""

from gpjax.linalg.operations import (
    diag,
    logdet,
    lower_cholesky,
    solve,
)
from gpjax.linalg.operators import (
    BlockDiag,
    Dense,
    Diagonal,
    Identity,
    Kronecker,
    LinearOperator,
    Triangular,
)
from gpjax.linalg.utils import (
    PSD,
    psd,
)

__all__ = [
    "LinearOperator",
    "Dense",
    "Diagonal",
    "Identity",
    "Triangular",
    "BlockDiag",
    "Kronecker",
    "lower_cholesky",
    "solve",
    "logdet",
    "diag",
    "psd",
    "PSD",
]
