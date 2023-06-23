"""Compute Random Fourier Feature (RFF) kernel approximations.  """
from dataclasses import dataclass

import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Float
import tensorflow_probability.substrates.jax.bijectors as tfb
import beartype.typing as tp

from gpjax.base import (
    param_field,
    static_field,
)
from gpjax.kernels.base import AbstractKernel
from gpjax.kernels.computations import (
    BasisFunctionComputation,
    NonStationaryBasisFunctionComputation,
)
from gpjax.typing import (
    Array,
    KeyArray,
    ScalarFloat
)


@dataclass
class RFF(AbstractKernel):
    r"""Computes an approximation of the kernel using Random Fourier Features.

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
    """

    base_kernel: AbstractKernel = None
    num_basis_fns: int = static_field(50)
    frequencies: Float[Array, "M 1"] = param_field(None, bijector=tfb.Identity())
    key: KeyArray = static_field(jr.PRNGKey(123))

    def __post_init__(self) -> None:
        r"""Post-initialisation function.

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
        self.name = f"{self.base_kernel.name} (RFF)"

    def __call__(self, x: Array, y: Array) -> Array:
        """Superfluous for RFFs."""

    def _check_valid_base_kernel(self, kernel: AbstractKernel):
        r"""Verify that the base kernel is valid for RFF approximation.

        Args:
            kernel (AbstractKernel): The kernel to be checked.
        """
        if kernel is None:
            raise ValueError("Base kernel must be specified.")
        error_msg = """
        Base kernel must have a spectral density. Currently, only Matérn
        and RBF kernels have implemented spectral densities.
        """
        if kernel.spectral_density is None:
            raise ValueError(error_msg)

    def compute_features(self, x: Float[Array, "N D"]) -> Float[Array, "N L"]:
        r"""Compute the features for the inputs.

        Args:
            x: A $`N \times D`$ array of inputs.

        Returns
        -------
            Float[Array, "N L"]: A $`N \times L`$ array of features where $`L = 2M`$.
        """
        return self.compute_engine(self).compute_features(x)


@dataclass
class NonStationaryRFF(AbstractKernel):
    """Kernel class for nonstationary Random Fourier Features."""

    num_basis_fns: int = static_field(None)
    n_dims: int = static_field(1)
    variance: Float[Array, ""] = param_field(jnp.array(1.0), bijector=tfb.Softplus())
    key: KeyArray = static_field(jr.PRNGKey(123), repr=False)
    lengthscale: tp.Union[ScalarFloat, Float[Array, " D"]] = param_field(
        jnp.array(1.0), bijector=tfb.Softplus()
    )
    frequencies: Float[Array, "2 M D"] = param_field(None, init=False, trainable=True)
    dropout: ScalarFloat = static_field(0.0)
    name: str = static_field("NonStationaryRFF", repr=False)
    compute_engine: tp.Type[NonStationaryBasisFunctionComputation] = None


    def __post_init__(self) -> None:
        self.compute_engine = NonStationaryBasisFunctionComputation
        self.frequencies = jr.normal(self.key, (2, self.num_basis_fns, self.n_dims))

    def __call__(self, x, y):
        return self.compute_engine(self).cross_covariance(x, y)

    def compute_features(self, x):
        return self.compute_engine(self).compute_features(x)
