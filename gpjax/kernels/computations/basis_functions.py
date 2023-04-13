from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, Float

from gpjax.linops import DenseLinearOperator

from .base import AbstractKernelComputation


@dataclass
class BasisFunctionComputation(AbstractKernelComputation):
    """Compute engine class for finite basis function approximations to a kernel."""

    num_basis_fns: int = None

    def cross_covariance(
        self, x: Float[Array, "N D"], y: Float[Array, "M D"]
    ) -> Float[Array, "N M"]:
        """For a pair of inputs, compute the cross covariance matrix between the inputs.
        Args:
            x: A N x D array of inputs.
            y: A M x D array of inputs.

        Returns:
            _type_: A N x M array of cross-covariances.
        """
        z1 = self.compute_features(x)
        z2 = self.compute_features(y)
        z1 /= self.kernel.num_basis_fns
        return self.kernel.base_kernel.variance * jnp.matmul(z1, z2.T)

    def gram(self, inputs: Float[Array, "N D"]) -> DenseLinearOperator:
        """For the Gram matrix, we can save computations by computing only one matrix multiplication between the inputs and the scaled frequencies.

        Args:
            inputs: A N x D array of inputs.

        Returns:
            DenseLinearOperator: A dense linear operator representing the N x N Gram matrix.
        """
        z1 = self.compute_features(inputs)
        matrix = jnp.matmul(z1, z1.T)  # shape: (n_samples, n_samples)
        matrix /= self.kernel.num_basis_fns
        return DenseLinearOperator(self.kernel.base_kernel.variance * matrix)

    def compute_features(self, x: Float[Array, "N D"]) -> Float[Array, "N L"]:
        """Compute the features for the inputs.

        Args:
            x: A N x D array of inputs.
            frequencies: A M x D array of frequencies.

        Returns:
            Float[Array, "N L"]: A N x L array of features where L = 2M.
        """
        frequencies = self.kernel.frequencies
        scaling_factor = self.kernel.base_kernel.lengthscale
        z = jnp.matmul(x, (frequencies / scaling_factor).T)
        z = jnp.concatenate([jnp.cos(z), jnp.sin(z)], axis=-1)
        return z
