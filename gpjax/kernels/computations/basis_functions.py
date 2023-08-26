from dataclasses import dataclass
import typing as tp

import jax.numpy as jnp
from jaxtyping import Float

from gpjax.kernels.computations.base import AbstractKernelComputation
from gpjax.typing import Array

Kernel = tp.TypeVar("Kernel", bound="gpjax.kernels.base.AbstractKernel")  # noqa: F821

from cola import PSD
from cola.ops import (
    Dense,
    LinearOperator,
)

# TODO: Use low rank linear operator!


@dataclass
class BasisFunctionComputation(AbstractKernelComputation):
    r"""Compute engine class for finite basis function approximations to a kernel."""

    def cross_covariance(
        self, kernel: Kernel, x: Float[Array, "N D"], y: Float[Array, "M D"]
    ) -> Float[Array, "N M"]:
        r"""Compute an approximate cross-covariance matrix.

        For a pair of inputs, compute the cross covariance matrix between the inputs.

        Args:
            kernel (Kernel): the kernel function.
            x: (Float[Array, "N D"]): A $`N \times D`$ array of inputs.
            y: (Float[Array, "M D"]): A $`M \times D`$ array of inputs.

        Returns:
            Float[Array, "N M"]: A $N \times M$ array of cross-covariances.
        """
        z1 = self.compute_features(kernel, x)
        z2 = self.compute_features(kernel, y)
        return self.scaling(kernel) * jnp.matmul(z1, z2.T)

    def gram(self, kernel: Kernel, inputs: Float[Array, "N D"]) -> LinearOperator:
        r"""Compute an approximate Gram matrix.

        For the Gram matrix, we can save computations by computing only one matrix
        multiplication between the inputs and the scaled frequencies.

        Args:
            kernel (Kernel): the kernel function.
            inputs (Float[Array, "N D"]): A $`N x D`$ array of inputs.

        Returns:
            LinearOperator: A dense linear operator representing the
                $`N \times N`$ Gram matrix.
        """
        z1 = self.compute_features(kernel, inputs)
        return PSD(Dense(self.scaling(kernel) * jnp.matmul(z1, z1.T)))

    def compute_features(
        self, kernel: Kernel, x: Float[Array, "N D"]
    ) -> Float[Array, "N L"]:
        r"""Compute the features for the inputs.

        Args:
            kernel (Kernel): the kernel function.
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

    def scaling(self, kernel: Kernel):
        r"""Compute the scaling factor for the covariance matrix.

        Args:
            kernel (Kernel): the kernel function.

        Returns
        -------
            Float[Array, ""]: A scalar array.
        """
        return kernel.base_kernel.variance / kernel.num_basis_fns
