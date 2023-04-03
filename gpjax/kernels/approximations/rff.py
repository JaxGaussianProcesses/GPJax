from dataclasses import dataclass
from typing import Any, Dict

import tensorflow_probability.substrates.jax as tfp
from jax.random import KeyArray, PRNGKey
from jaxtyping import Array, Float
from simple_pytree import static_field

from ...parameters import param_field
from ..base import AbstractKernel
from ..computations import (
    AbstractKernelComputation,
    BasisFunctionComputation,
    DenseKernelComputation,
)

tfb = tfp.bijectors


@dataclass
class AbstractFourierKernel:
    base_kernel: AbstractKernel
    num_basis_fns: int
    frequencies: Float[Array, "M 1"] = param_field(None, bijector=tfb.Identity)
    key: KeyArray = static_field(PRNGKey(123))


@dataclass
class RFF(AbstractKernel, AbstractFourierKernel):
    """Computes an approximation of the kernel using Random Fourier Features.

    All stationary kernels are equivalent to the Fourier transform of a probability
    distribution. We call the corresponding distribution the spectral density. Using
    a finite number of basis functions, we can compute the spectral density using a
    Monte-Carlo approximation. This is done by sampling from the spectral density and
    computing the Fourier transform of the samples. The kernel is then approximated by
    the inner product of the Fourier transform of the samples with the Fourier
    transform of the data.

    The key reference for this implementation is the following papers:
    - 'Random Features for Large-Scale Kernel Machines' by Rahimi and Recht (2008).
    - 'On the Error of Random Fourier Features' by Sutherland and Schneider (2015).

    Args:
        AbstractKernel (_type_): _description_
    """

    def __post_init__(self) -> None:
        """Post-initialisation function.

        This function is called after the initialisation of the kernel. It is used to
        set the computation engine to be the basis function computation engine.
        """
        self._check_valid_base_kernel(self.base_kernel)
        self.compute_engine = BasisFunctionComputation

        if self.frequencies is None:
            n_dims = self.base_kernel.ndims
            self.frequencies = self.base_kernel.spectral_density.sample(
                seed=self.key, sample_shape=(self.num_basis_fns, n_dims)
            )

    def __call__(self, x: Array, y: Array) -> Array:
        pass

    def _check_valid_base_kernel(self, kernel: AbstractKernel):
        """Verify that the base kernel is valid for RFF approximation.

        Args:
            kernel (AbstractKernel): The kernel to be checked.
        """
        error_msg = """
        Base kernel must have a spectral density. Currently, only Matérn
        and RBF kernels have implemented spectral densities.
        """
        if kernel.spectral_density is None:
            raise ValueError(error_msg)
