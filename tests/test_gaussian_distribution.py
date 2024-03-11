# %% [markdown]
# Copyright 2022 The Jax Linear Operator Contributors All Rights Reserved.
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


from jax import config
import jax.numpy as jnp
import jax.random as jr
import pytest

# Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)

import cola
from cola.ops import (
    Dense,
    Diagonal,
)

from gpjax.distributions import GaussianDistribution

_key = jr.key(seed=42)

from tensorflow_probability.substrates.jax.distributions import (
    MultivariateNormalDiag,
    MultivariateNormalFullCovariance,
)


def approx_equal(res: jnp.ndarray, actual: jnp.ndarray) -> bool:
    """Check if two arrays are approximately equal."""
    return jnp.linalg.norm(res - actual) < 1e-5


@pytest.mark.parametrize("n", [1, 2, 5, 100])
def test_array_arguments(n: int) -> None:
    key_mean, key_sqrt = jr.split(_key, 2)
    mean = jr.uniform(key_mean, shape=(n,))
    sqrt = jr.uniform(key_sqrt, shape=(n, n))
    covariance = sqrt @ sqrt.T
    # check that cholesky does not error
    _L = jnp.linalg.cholesky(covariance)  # noqa: F841

    dist = GaussianDistribution(loc=mean, scale=cola.PSD(Dense(covariance)))

    assert approx_equal(dist.mean(), mean)
    assert approx_equal(dist.variance(), covariance.diagonal())
    assert approx_equal(dist.stddev(), jnp.sqrt(covariance.diagonal()))
    assert approx_equal(dist.covariance(), covariance)

    assert isinstance(dist.scale, Dense)
    assert cola.PSD in dist.scale.annotations

    y = jr.uniform(_key, shape=(n,))

    tfp_dist = MultivariateNormalFullCovariance(loc=mean, covariance_matrix=covariance)

    assert approx_equal(dist.log_prob(y), tfp_dist.log_prob(y))
    assert approx_equal(dist.kl_divergence(dist), 0.0)


@pytest.mark.parametrize("n", [1, 2, 5, 100])
def test_diag_linear_operator(n: int) -> None:
    key_mean, key_diag = jr.split(_key, 2)
    mean = jr.uniform(key_mean, shape=(n,))
    diag = jr.uniform(key_diag, shape=(n,))

    # We purosely forget to add a PSD annotation to the diagonal matrix.
    dist_diag = GaussianDistribution(loc=mean, scale=Diagonal(diag**2))
    tfp_dist = MultivariateNormalDiag(loc=mean, scale_diag=diag)

    # We check that the PSD annotation is added automatically.
    assert isinstance(dist_diag.scale, Diagonal)
    assert cola.PSD in dist_diag.scale.annotations

    assert approx_equal(dist_diag.mean(), tfp_dist.mean())
    assert approx_equal(dist_diag.mode(), tfp_dist.mode())
    assert approx_equal(dist_diag.entropy(), tfp_dist.entropy())
    assert approx_equal(dist_diag.variance(), tfp_dist.variance())
    assert approx_equal(dist_diag.stddev(), tfp_dist.stddev())
    assert approx_equal(dist_diag.covariance(), tfp_dist.covariance())

    gpjax_samples = dist_diag.sample(seed=_key, sample_shape=(10,))
    tfp_samples = tfp_dist.sample(seed=_key, sample_shape=(10,))
    assert approx_equal(gpjax_samples, tfp_samples)

    y = jr.uniform(_key, shape=(n,))

    assert approx_equal(dist_diag.log_prob(y), tfp_dist.log_prob(y))
    assert approx_equal(dist_diag.log_prob(y), tfp_dist.log_prob(y))

    assert approx_equal(dist_diag.kl_divergence(dist_diag), 0.0)


@pytest.mark.parametrize("n", [1, 2, 5, 100])
def test_dense_linear_operator(n: int) -> None:
    key_mean, key_sqrt = jr.split(_key, 2)
    mean = jr.uniform(key_mean, shape=(n,))
    sqrt = jr.uniform(key_sqrt, shape=(n, n))
    covariance = sqrt @ sqrt.T

    sqrt = jnp.linalg.cholesky(covariance + jnp.eye(n) * 1e-10)

    dist_dense = GaussianDistribution(loc=mean, scale=cola.PSD(Dense(covariance)))
    tfp_dist = MultivariateNormalFullCovariance(loc=mean, covariance_matrix=covariance)

    assert approx_equal(dist_dense.mean(), tfp_dist.mean())
    assert approx_equal(dist_dense.mode(), tfp_dist.mode())
    assert approx_equal(dist_dense.entropy(), tfp_dist.entropy())
    assert approx_equal(dist_dense.variance(), tfp_dist.variance())
    assert approx_equal(dist_dense.stddev(), tfp_dist.stddev())
    assert approx_equal(dist_dense.covariance(), tfp_dist.covariance())

    assert approx_equal(
        dist_dense.sample(seed=_key, sample_shape=(10,)),
        tfp_dist.sample(seed=_key, sample_shape=(10,)),
    )

    y = jr.uniform(_key, shape=(n,))

    assert approx_equal(dist_dense.log_prob(y), tfp_dist.log_prob(y))
    assert approx_equal(dist_dense.kl_divergence(dist_dense), 0.0)


@pytest.mark.parametrize("n", [1, 2, 5, 100])
def test_kl_divergence(n: int) -> None:
    key_a, key_b = jr.split(_key, 2)
    mean_a = jr.uniform(key_a, shape=(n,))
    mean_b = jr.uniform(key_b, shape=(n,))
    sqrt_a = jr.uniform(key_a, shape=(n, n))
    sqrt_b = jr.uniform(key_b, shape=(n, n))
    covariance_a = sqrt_a @ sqrt_a.T
    covariance_b = sqrt_b @ sqrt_b.T

    dist_a = GaussianDistribution(loc=mean_a, scale=cola.PSD(Dense(covariance_a)))
    dist_b = GaussianDistribution(loc=mean_b, scale=cola.PSD(Dense(covariance_b)))

    tfp_dist_a = MultivariateNormalFullCovariance(
        loc=mean_a, covariance_matrix=covariance_a
    )
    tfp_dist_b = MultivariateNormalFullCovariance(
        loc=mean_b, covariance_matrix=covariance_b
    )

    assert approx_equal(
        dist_a.kl_divergence(dist_b), tfp_dist_a.kl_divergence(tfp_dist_b)
    )

    with pytest.raises(ValueError):
        incompatible = GaussianDistribution(loc=jnp.ones((2 * n,)))
        incompatible.kl_divergence(dist_a)
