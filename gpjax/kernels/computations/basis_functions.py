from dataclasses import dataclass

import jax.numpy as jnp
import jax.random as jr
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
        z1 /= self.kernel.num_basis_fns
        return self.kernel.base_kernel.variance * jnp.matmul(z1, z2.T)

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
        matrix = jnp.matmul(z1, z1.T)  # shape: (n_samples, n_samples)
        matrix /= self.kernel.num_basis_fns
        return DenseLinearOperator(self.kernel.base_kernel.variance * matrix)

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


@dataclass
class NonStationaryBasisFunctionComputation(AbstractKernelComputation):
    """Finite basis function approximations to a nonstationary kernel."""

    key = jr.PRNGKey(123)

    def cross_covariance(
        self, x: Float[Array, "N D"], y: Float[Array, "N_ D"]
    ) -> Float[Array, "N N_"]:
        r"""Compute a cross-covariance matrix.

        For a pair of inputs, compute the cross covariance matrix between the inputs.

        Args:
            x: (Float[Array, "N D"]): A $`N \times D`$ array of inputs.
            y: (Float[Array, "M D"]): A $`M \times D`$ array of inputs.

        Returns:
            Float[Array, "N M"]: A $N \times M$ array of cross-covariances.
        """
        Ω_1, Ω_2 = self.kernel.frequencies

        if d := self.kernel.dropout:
            Ω_1, Ω_2 = self._gaussian_dropout(Ω_1, Ω_2, d)

        Φ_x = self.compute_features(x, Ω_1, Ω_2)
        Φ_y = self.compute_features(y, Ω_1, Ω_2)
        return (self.scaling * jnp.matmul(Φ_x, Φ_y.T)).astype(jnp.float64)

    def gram(self, inputs: Float[Array, "N D"]) -> DenseLinearOperator:
        r"""Compute a Gram matrix.

        For the Gram matrix, we can save computations by computing only one matrix
        multiplication between the inputs and the scaled frequencies.

        Args:
            inputs (Float[Array, "N D"]): A $`N x D`$ array of inputs.

        Returns:
            DenseLinearOperator: A dense linear operator representing the
                $`N \times N`$ Gram matrix.
        """
        Ω_1, Ω_2 = self.kernel.frequencies

        if d := self.kernel.dropout:
            Ω_1, Ω_2 = self._gaussian_dropout(Ω_1, Ω_2, d)

        Φ_x = self.compute_features(inputs, Ω_1, Ω_2)
        return DenseLinearOperator(
            self.scaling * jnp.matmul(Φ_x, Φ_x.T), dtype=jnp.float64
        )

    def compute_features(
        self, x: Float[Array, "N D"], Ω_1, Ω_2
    ) -> Float[Array, "N 2*M"]:
        r"""Compute the features for the inputs.

        Args:
            x (Float[Array, "N D"]): A $`N \times D`$ array of inputs.

        Returns
        -------
            Float[Array, "N L"]: A $`N \times L`$ array of features where $`L = 2M`$.
        """
        x = self.kernel.slice_input(x)
        z_1 = jnp.matmul(x, (Ω_1 / self.kernel.lengthscale).T)
        z_2 = jnp.matmul(x, (Ω_2 / self.kernel.lengthscale).T)
        Φ = jnp.concatenate(
            [jnp.cos(z_1) + jnp.cos(z_2), jnp.sin(z_1) + jnp.sin(z_2)], axis=-1
        )
        return Φ

    @property
    def scaling(self):
        return self.kernel.variance / (4 * self.kernel.num_basis_fns)

    def _gaussian_dropout(self, Ω_1, Ω_2, d):
        self.key, subkey = jr.split(self.key)
        Ω_1 += jr.normal(self.key, Ω_1.shape, dtype=Ω_1.dtype) * d
        Ω_2 += jr.normal(subkey, Ω_2.shape, dtype=Ω_2.dtype) * d
        return Ω_1, Ω_2
