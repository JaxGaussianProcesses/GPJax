from gpjax.parameters import log_density
from tensorflow_probability.substrates.jax import distributions as tfd
import pytest
import jax.numpy as jnp


@pytest.mark.parametrize('x', [-1., 0., 1.])
def test_lpd(x):
    val = jnp.array(x)
    dist = tfd.Normal(loc=0., scale=1.)
    lpd = log_density(val, dist)
    assert lpd is not None