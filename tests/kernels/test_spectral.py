import jax.numpy as jnp
import jax.random as jr
import pytest
from tensorflow_probability.substrates.jax import distributions as tfd

from gpjax.kernels import RBF
from gpjax.kernels.spectral import (
    SpectralRBF,
    initialise,
    sample_frequencies,
    spectral_density,
    to_spectral,
)


@pytest.mark.parametrize("n_basis", [1, 2, 10])
def test_initialise(n_basis):
    key = jr.PRNGKey(123)
    kernel = SpectralRBF(num_basis=n_basis)
    params = initialise(key, kernel)
    assert list(params.keys()) == ["basis_fns", "lengthscale", "variance"]
    for v in params.values():
        assert v.dtype == jnp.float64


@pytest.mark.parametrize("n_basis", [1, 2, 10])
def test_to_spectral(n_basis):
    base_kern = RBF()
    spectral = to_spectral(base_kern, n_basis)
    assert isinstance(spectral, SpectralRBF)
    assert spectral.num_basis == n_basis
    assert spectral.stationary


def test_spectral_density():
    kernel = SpectralRBF(num_basis=10)
    sdensity = spectral_density(kernel)
    assert isinstance(sdensity, tfd.Normal)


@pytest.mark.parametrize("n_freqs", [1, 2, 5])
def test_sample_frequencies(n_freqs):
    key = jr.PRNGKey(123)
    kernel = SpectralRBF(num_basis=n_freqs)
    sdensity = spectral_density(kernel)
    omega = sample_frequencies(key, kernel, n_freqs, 1)
    omegad = sample_frequencies(key, sdensity, n_freqs, 1)
    assert (omegad == omega).all()
    assert omegad.dtype == jnp.float64
    assert omega.dtype == jnp.float64
