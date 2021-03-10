import jax.numpy as jnp
from multipledispatch import dispatch
from gpjax.kernels import RBF
from tensorflow_probability.substrates.jax import distributions as tfd
from chex import dataclass
from .base import Kernel
from ..types import Array
from ..utils import sort_dictionary, I
from .utils import stretch, scale
from typing import Optional


@dataclass(repr=False)
class SpectralKernel:
    num_basis: int


@dataclass(repr=False)
class SpectralRBF(Kernel, SpectralKernel):
    name: Optional[str] = 'Spectral RBF'
    stationary: str = True

    def __repr__(self):
        return f"{self.name}:\n\t Number of basis functions: {self.num_basis}\n\t Stationary: {self.stationary} \n\t ARD structure: {self.ard}"

    def __call__(self, x: jnp.DeviceArray, y: jnp.DeviceArray, params: dict) -> Array:
        phi = self._build_phi(x, params)
        return jnp.matmul(phi, jnp.transpose(phi))/self.num_basis

    def _build_phi(self, x: jnp.DeviceArray, params):
        scaled_freqs = scale(params['basis_fns'], params['lengthscale'])
        phi = jnp.matmul(x, jnp.transpose(scaled_freqs))
        return jnp.hstack([jnp.cos(phi), jnp.sin(phi)])

@dispatch(RBF, int)
def to_spectral(kernel: RBF, num_basis: int):
    return SpectralRBF(num_basis = num_basis)


@dispatch(SpectralRBF)
def spectral_density(kernel: SpectralRBF) -> tfd.Distribution:
    return tfd.Normal(loc = jnp.array(0.), scale=jnp.array(1.))


@dispatch(jnp.DeviceArray, tfd.Distribution, int, int)
def sample_frequencies(key, density: tfd.Distribution, n_frequencies: int, input_dimension: int) -> jnp.DeviceArray:
    return density.sample(sample_shape=(n_frequencies, input_dimension), seed=key)


@dispatch(jnp.DeviceArray, SpectralKernel, int, int)
def sample_frequencies(key, kernel: SpectralKernel, n_frequencies: int, input_dimension: int) -> jnp.DeviceArray:
    density = spectral_density(kernel)
    return density.sample(sample_shape=(n_frequencies, input_dimension), seed=key)


@dispatch(jnp.DeviceArray, SpectralRBF)
def initialise(key: jnp.DeviceArray, kernel: SpectralRBF):
    basis_init = sample_frequencies(key, kernel, kernel.num_basis, kernel.ndims)
    return {"basis_fns": basis_init, "lengthscale": jnp.array([1.0] * kernel.ndims), "variance": jnp.array([1.0])}
