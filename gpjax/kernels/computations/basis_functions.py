from typing import Callable, Dict

import jax.numpy as jnp
from jaxtyping import Array, Float
from .base import AbstractKernelComputation
from jaxlinop import DenseLinearOperator


class BasisFunctionComputation(AbstractKernelComputation):
    """Compute engine class for finite basis function approximations to a kernel."""

    def __init__(
        self,
        kernel_fn: Callable[
            [Dict, Float[Array, "1 D"], Float[Array, "1 D"]], Array
        ] = None,
    ) -> None:
        """Initialise the computation engine for a basis function approximation to a kernel.

        Args:
            kernel_fn: A. The kernel function for which the compute engine is assigned to.
        """
        super().__init__(kernel_fn)
        self._num_basis_fns = None

    @property
    def num_basis_fns(self) -> float:
        """The number of basis functions used to approximate the kernel."""
        return self._num_basis_fns

    @num_basis_fns.setter
    def num_basis_fns(self, num_basis_fns: int) -> None:
        self._num_basis_fns = float(num_basis_fns)

    def cross_covariance(
        self, params: Dict, x: Float[Array, "N D"], y: Float[Array, "M D"]
    ) -> Float[Array, "N M"]:
        """For a pair of inputs, compute the cross covariance matrix between the inputs.
        Args:
            params (Dict): A dictionary of parameters for which the cross-covariance matrix should be constructed with.
            x: A N x D array of inputs.
            y: A M x D array of inputs.

        Returns:
            _type_: A N x M array of cross-covariances.
        """
        z1 = self.compute_features(
            x, params["frequencies"], scaling_factor=params["lengthscale"]
        )
        z2 = self.compute_features(
            y, params["frequencies"], scaling_factor=params["lengthscale"]
        )
        z1 /= self.num_basis_fns
        return params["variance"] * jnp.matmul(z1, z2.T)

    def gram(self, params: Dict, inputs: Float[Array, "N D"]) -> DenseLinearOperator:
        """For the Gram matrix, we can save computations by computing only one matrix multiplication between the inputs and the scaled frequencies.

        Args:
            params (Dict): A dictionary of parameters for which the Gram matrix should be constructed with.
            inputs: A N x D array of inputs.

        Returns:
            DenseLinearOperator: A dense linear operator representing the N x N Gram matrix.
        """
        z1 = self.compute_features(
            inputs, params["frequencies"], scaling_factor=params["lengthscale"]
        )
        matrix = jnp.matmul(z1, z1.T)  # shape: (n_samples, n_samples)
        matrix /= self.num_basis_fns
        return DenseLinearOperator(params["variance"] * matrix)

    @staticmethod
    def compute_features(
        x: Float[Array, "N D"],
        frequencies: Float[Array, "M D"],
        scaling_factor: Float[Array, "D"] = None,
    ) -> Float[Array, "N L"]:
        """Compute the features for the inputs.

        Args:
            x: A N x D array of inputs.
            frequencies: A M x D array of frequencies.

        Returns:
            Float[Array, "N L"]: A N x L array of features where L = 2M.
        """
        if scaling_factor is not None:
            frequencies = frequencies / scaling_factor
        z = jnp.matmul(x, frequencies.T)
        z = jnp.concatenate([jnp.cos(z), jnp.sin(z)], axis=-1)
        return z
