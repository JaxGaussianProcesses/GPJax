from typing import Optional

import jax.numpy as jnp
from chex import dataclass
from multipledispatch import dispatch
from tensorflow_probability.substrates.jax import distributions as tfd
import abc
from gpjax.kernels import RBF

from ..types import Array
from ..utils import I, sort_dictionary
from .base import Kernel
from ..config import get_defaults


@dataclass(repr=False)
class SpectralKernel:
    num_basis: int

    @property
    @abc.abstractmethod
    def spectral_density(self) -> tfd.Distribution:
        raise NotImplementedError


@dataclass(repr=False)
class SpectralRBF(Kernel, SpectralKernel):
    name: Optional[str] = "Spectral RBF"
    stationary: str = True

    def __repr__(self):
        return f"{self.name}:\n\t Number of basis functions: {self.num_basis}\n\t Stationary: {self.stationary} \n\t ARD structure: {self.ard}"

    def __call__(self, x: jnp.DeviceArray, y: jnp.DeviceArray, params: dict) -> Array:
        phi = self._build_phi(x, params)
        return jnp.matmul(phi, jnp.transpose(phi)) / self.num_basis

    def _build_phi(self, x: jnp.DeviceArray, params):
        scaled_freqs = params["basis_fns"]/ params["lengthscale"]
        phi = jnp.matmul(x, jnp.transpose(scaled_freqs))
        return jnp.hstack([jnp.cos(phi), jnp.sin(phi)])

    @property
    def spectral_density(self) -> tfd.Distribution:
        return tfd.Normal(loc=jnp.array(0.0), scale=jnp.array(1.0))

    @property
    def params(self) -> dict:
        default_config = get_defaults()
        key = default_config.key
        initial_frequencies = self.spectral_density.sample(sample_shape=(self.num_basis, self.input_dimension), seed = key)
        return {
            "basis_fns": initial_frequencies,
            "lengthscale": jnp.array([1.0] * self.ndims),
            "variance": jnp.array([1.0]),
        }


def to_spectral(kernel: RBF, num_basis: int):
    if isinstance(kernel, RBF):
        return SpectralRBF(num_basis=num_basis)
    else:
        raise NotImplementedError(f'No spectral kernel implemented for {kernel.name}')
