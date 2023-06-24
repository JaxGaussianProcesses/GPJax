from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Float

from gpjax.kernels.computations.base import AbstractKernelComputation
from gpjax.linops import DenseLinearOperator
from gpjax.typing import Array


@dataclass
class BasisFunctionComputation(AbstractKernelComputation):
    r"""Compute engine class for finite basis function approximations to a kernel."""

    num_basis_fns: int = None

    def cross_covariance(
        self, x: Float[Array, "N D"], y: Float[Array, "M D"]
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
        return self.scaling * jnp.matmul(z1, z2.T)

    def gram(self, inputs: Float[Array, "N D"]) -> DenseLinearOperator:
        r"""Compute an approximate Gram matrix.

        For the Gram matrix, we can save computations by computing only one matrix
        multiplication between the inputs and the scaled frequencies.

        Args:
            inputs (Float[Array, "N D"]): A $`N x D`$ array of inputs.

        Returns:
            DenseLinearOperator: A dense linear operator representing the
                $`N \times N`$ Gram matrix.
        """
        z1 = self.compute_features(inputs)
        return DenseLinearOperator(self.scaling * jnp.matmul(z1, z1.T))

    def compute_features(self, x: Float[Array, "N D"]) -> Float[Array, "N L"]:
        r"""Compute the features for the inputs.

        Args:
            x (Float[Array, "N D"]): A $`N \times D`$ array of inputs.

        Returns
        -------
            Float[Array, "N L"]: A $`N \times L`$ array of features where $`L = 2M`$.
        """
        frequencies = self.kernel.frequencies
        scaling_factor = self.kernel.base_kernel.lengthscale
        z = jnp.matmul(x, (frequencies / scaling_factor).T)
        z = jnp.concatenate([jnp.cos(z), jnp.sin(z)], axis=-1)
        return z

    @property
    def scaling(self):
        return self.kernel.base_kernel.variance / self.kernel.num_basis_fns
