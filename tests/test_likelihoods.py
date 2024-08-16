# Copyright 2022 The GPJax Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from typing import (
    Callable,
    Tuple,
)

from jax import config
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import (
    Array,
    Float,
)
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

from gpjax.likelihoods import (
    Bernoulli,
    Gaussian,
    Poisson,
    inv_probit,
)

# Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)
_initialise_key = jr.key(123)


def _compute_latent_dist(
    n: int,
) -> Tuple[
    tfd.MultivariateNormalFullCovariance, Float[Array, " N"], Float[Array, "N N"]
]:
    k1, k2 = jr.split(_initialise_key)
    latent_mean = jr.uniform(k1, shape=(n,))
    latent_sqrt = jr.uniform(k2, shape=(n, n))
    latent_cov = jnp.matmul(latent_sqrt, latent_sqrt.T)
    latent_dist = tfd.MultivariateNormalFullCovariance(latent_mean, latent_cov)
    return latent_dist, latent_mean, latent_cov


@pytest.mark.parametrize("n", [1, 2, 10])
@pytest.mark.parametrize("obs_stddev", [0.1, 0.5, 1.0])
def test_gaussian_likelihood(n: int, obs_stddev: float):
    x = jnp.linspace(-3.0, 3.0).reshape(-1, 1)
    likelihood = Gaussian(num_datapoints=n, obs_stddev=obs_stddev)

    assert isinstance(likelihood.link_function, Callable)
    assert isinstance(likelihood.link_function(x), tfd.Distribution)

    # Construct latent function distribution.
    latent_dist, latent_mean, latent_cov = _compute_latent_dist(n)
    pred_dist = likelihood(latent_dist)
    assert isinstance(pred_dist, tfd.MultivariateNormalFullCovariance)

    # Check predictive mean and variance.
    assert (pred_dist.mean() == latent_mean).all()
    noise_matrix = jnp.eye(likelihood.num_datapoints) * likelihood.obs_stddev.value**2
    assert np.allclose(
        pred_dist.scale_tril, jnp.linalg.cholesky(latent_cov + noise_matrix)
    )


@pytest.mark.parametrize("n", [1, 2, 10])
def test_bernoulli_likelihood(n: int):
    x = jnp.linspace(-3.0, 3.0).reshape(-1, 1)
    likelihood = Bernoulli(num_datapoints=n)

    assert isinstance(likelihood.link_function, Callable)
    assert isinstance(likelihood.link_function(x), tfd.Distribution)

    # Construct latent function distribution.
    latent_dist, latent_mean, latent_cov = _compute_latent_dist(n)
    pred_dist = likelihood(latent_dist)
    assert isinstance(pred_dist, tfd.Bernoulli)

    # Check predictive mean and variance.
    p = inv_probit(latent_mean / jnp.sqrt(1.0 + jnp.diagonal(latent_cov)))
    assert (pred_dist.mean() == p).all()
    assert (pred_dist.variance() == p * (1.0 - p)).all()


@pytest.mark.parametrize("n", [1, 2, 10])
def test_poisson_likelihood(n: int):
    x = jnp.linspace(-3.0, 3.0).reshape(-1, 1)
    likelihood = Poisson(num_datapoints=n)

    assert isinstance(likelihood.link_function, Callable)
    assert isinstance(likelihood.link_function(x), tfd.Distribution)

    # Construct latent function distribution.
    latent_dist, latent_mean, latent_cov = _compute_latent_dist(n)
    pred_dist = likelihood(latent_dist)
    assert isinstance(pred_dist, tfd.Poisson)

    # Check predictive mean and variance.
    rate = jnp.exp(latent_mean)
    assert (pred_dist.mean() == rate).all()
