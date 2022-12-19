"""JaxKern."""
from .kernels import (
    RBF,
    GraphKernel,
    Matern12,
    Matern32,
    Matern52,
    Polynomial,
    ProductKernel,
    SumKernel,
    DenseKernelComputation,
    DiagonalKernelComputation,
    ConstantDiagonalKernelComputation,
)

__all__ = [
    "RBF",
    "GraphKernel",
    "Matern12",
    "Matern32",
    "Matern52",
    "Polynomial",
    "ProductKernel",
    "SumKernel",
    "DenseKernelComputation",
    "DiagonalKernelComputation",
    "ConstantDiagonalKernelComputation",
]

from . import _version

__version__ = _version.get_versions()["version"]
