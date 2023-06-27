from dataclasses import dataclass
import typing as tp

import jax.numpy as jnp
from jaxtyping import (
    Array,
    Float,
)

from gpjax.kernels.computations.base import AbstractKernelComputation
from gpjax.linops import DenseLinearOperator

Kernel = tp.TypeVar("Kernel", bound="gpjax.kernels.base.AbstractKernel")  # noqa: F821


@dataclass
class BasisFunctionComputation(AbstractKernelComputation):
    r"""Compute engine class for finite basis function approximations to a kernel."""

    def cross_covariance(
        self, kernel: Kernel, x: Float[Array, "N D"], y: Float[Array, "M D"]
    ) -> Float[Array, "N M"]:
        r"""Compute an approximate cross-covariance matrix.

        For a pair of inputs, compute the cross covariance matrix between the inputs.

        Args:
            x: (Float[Array, "N D"]): A $`N \times D`$ array of inputs.
            y: (Float[Array, "M D"]): A $`M \times D`$ array of inputs.

        Returns:
            Float[Array, "N M"]: A $N \times M$ array of cross-covariances.
        """
        z1 = self.compute_features(x)
        z2 = self.compute_features(y)
        z1 /= kernel.num_basis_fns
        return kernel.base_kernel.variance * jnp.matmul(z1, z2.T)

    def gram(self, kernel: Kernel, inputs: Float[Array, "N D"]) -> DenseLinearOperator:
        r"""Compute an approximate Gram matrix.

        For the Gram matrix, we can save computations by computing only one matrix
        multiplication between the inputs and the scaled frequencies.

        Args:
            inputs (Float[Array, "N D"]): A $`N x D`$ array of inputs.

        Returns:
            DenseLinearOperator: A dense linear operator representing the
                $`N \times N`$ Gram matrix.
        """
        z1 = self.compute_features(kernel, inputs)
        matrix = jnp.matmul(z1, z1.T)  # shape: (n_samples, n_samples)
        matrix /= kernel.num_basis_fns
        return DenseLinearOperator(kernel.base_kernel.variance * matrix)

    def compute_features(
        self, kernel: Kernel, x: Float[Array, "N D"]
    ) -> Float[Array, "N L"]:
        r"""Compute the features for the inputs.

        Args:
            x (Float[Array, "N D"]): A $`N \times D`$ array of inputs.

        Returns
        -------
            Float[Array, "N L"]: A $`N \times L`$ array of features where $`L = 2M`$.
        """
        frequencies = kernel.frequencies
        scaling_factor = kernel.base_kernel.lengthscale
        z = jnp.matmul(x, (frequencies / scaling_factor).T)
        z = jnp.concatenate([jnp.cos(z), jnp.sin(z)], axis=-1)
        return z
