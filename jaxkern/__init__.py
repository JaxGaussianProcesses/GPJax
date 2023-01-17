"""JaxKern."""
from .base import ProductKernel, SumKernel
from .computations import (
    ConstantDiagonalKernelComputation,
    DenseKernelComputation,
    DiagonalKernelComputation,
    EigenKernelComputation,
)
from .nonstationary import (
    Linear,
    Periodic,
    Polynomial,
)
from .stationary import (
    RBF,
    Matern12,
    Matern32,
    Matern52,
    RationalQuadratic,
    PoweredExponential,
)
from .non_euclidean import GraphKernel

__all__ = [
    "RBF",
    "GraphKernel",
    "Matern12",
    "Matern32",
    "Matern52",
    "Linear",
    "Polynomial",
    "ProductKernel",
    "SumKernel",
    "DenseKernelComputation",
    "DiagonalKernelComputation",
    "ConstantDiagonalKernelComputation",
    "EigenKernelComputation",
    "PoweredExponential",
    "Periodic",
    "RationalQuadratic",
]

from . import _version

__version__ = _version.get_versions()["version"]
