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

from typing import Callable, Dict

import distrax as dx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax.random import KeyArray
from jax.config import config
from jaxtyping import Array, Float

from gpjax.likelihoods import (
    AbstractLikelihood,
    Bernoulli,
    Conjugate,
    Gaussian,
    NonConjugate,
    inv_probit,
)


# Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)
_initialise_key = jr.PRNGKey(123)

# Likelihood parameter names to test in initialisation.
true_initialisation = {
    "Gaussian": ["obs_noise"],
    "Bernoulli": [],
}


def test_abstract_likelihood():
    # Test that abstract likelihoods cannot be instantiated.
    with pytest.raises(TypeError):
        AbstractLikelihood(num_datapoints=123)

    # Create a dummy likelihood class with abstract methods implemented.
    class DummyLikelihood(AbstractLikelihood):
        def init_params(self, key: KeyArray) -> Dict:
            return {}

        def predict(self, params: Dict, dist: dx.Distribution) -> dx.Distribution:
            return dx.Normal(0.0, 1.0)

        def link_function(self) -> Callable:
            def link(x: Float[Array, "N 1"]) -> Float[Array, "N 1"]:
                return dx.MultivariateNormalDiag(loc=x)

            return link

    # Test that the dummy likelihood can be instantiated.
    dummy_likelihood = DummyLikelihood(num_datapoints=123)
    assert isinstance(dummy_likelihood, AbstractLikelihood)


@pytest.mark.parametrize("n", [1, 10])
@pytest.mark.parametrize("lik", [Gaussian, Bernoulli])
def test_initialisers(n: int, lik: AbstractLikelihood) -> None:
    key = _initialise_key

    # Initialise the likelihood.
    likelihood = lik(num_datapoints=n)

    # Get default parameter dictionary.
    params = likelihood.init_params(key)

    # Check parameter dictionary
    assert list(params.keys()) == true_initialisation[likelihood.name]
    assert len(list(params.values())) == len(true_initialisation[likelihood.name])


@pytest.mark.parametrize("n", [1, 10])
def test_bernoulli_predictive_moment(n: int) -> None:
    key = _initialise_key

    # Initialise bernoulli likelihood.
    likelihood = Bernoulli(num_datapoints=n)

    # Initialise parameters.
    params = likelihood.init_params(key)

    # Construct latent function mean and variance values
    mean_key, var_key = jr.split(key)
    fmean = jr.uniform(mean_key, shape=(n, 1))
    fvar = jnp.exp(jr.normal(var_key, shape=(n, 1)))

    # Test predictive moments.
    assert isinstance(likelihood.predictive_moment_fn, Callable)

    y = likelihood.predictive_moment_fn(params, fmean, fvar)
    y_mean = y.mean()
    y_var = y.variance()

    assert y_mean.shape == (n, 1)
    assert y_var.shape == (n, 1)


@pytest.mark.parametrize("lik", [Gaussian, Bernoulli])
@pytest.mark.parametrize("n", [1, 10])
def test_link_fns(lik: AbstractLikelihood, n: int) -> None:
    key = _initialise_key

    # Create test inputs.
    x = jnp.linspace(-3.0, 3.0).reshape(-1, 1)

    # Initialise likelihood.
    likelihood = lik(num_datapoints=n)

    # Initialise parameters.
    params = likelihood.init_params(key)

    # Test likelihood link function.
    assert isinstance(likelihood.link_function, Callable)
    assert isinstance(likelihood.link_function(params, x), dx.Distribution)


@pytest.mark.parametrize("noise", [0.1, 0.5, 1.0])
@pytest.mark.parametrize("n", [1, 2, 10])
def test_call_gaussian(noise: float, n: int) -> None:
    key = _initialise_key

    # Initialise likelihood and parameters.
    likelihood = Gaussian(num_datapoints=n)
    params = {"likelihood": {"obs_noise": noise}}

    # Construct latent function distribution.
    latent_mean = jr.uniform(key, shape=(n,))
    latent_sqrt = jr.uniform(key, shape=(n, n))
    latent_cov = jnp.matmul(latent_sqrt, latent_sqrt.T)
    latent_dist = dx.MultivariateNormalFullCovariance(latent_mean, latent_cov)

    # Test call method.
    pred_dist = likelihood(params, latent_dist)

    # Check that the distribution is a MultivariateNormalFullCovariance.
    assert isinstance(pred_dist, dx.MultivariateNormalFullCovariance)

    # Check predictive mean and variance.
    assert (pred_dist.mean() == latent_mean).all()

    noise_matrix = jnp.eye(n) * noise
    assert np.allclose(
        pred_dist.scale_tri, jnp.linalg.cholesky(latent_cov + noise_matrix)
    )


@pytest.mark.parametrize("n", [1, 2, 10])
def test_call_bernoulli(n: int) -> None:
    key = _initialise_key

    # Initialise likelihood and parameters.
    likelihood = Bernoulli(num_datapoints=n)
    params = {"likelihood": {}}

    # Construct latent function distribution.
    latent_mean = jr.uniform(key, shape=(n,))
    latent_sqrt = jr.uniform(key, shape=(n, n))
    latent_cov = jnp.matmul(latent_sqrt, latent_sqrt.T)
    latent_dist = dx.MultivariateNormalFullCovariance(latent_mean, latent_cov)

    # Test call method.
    pred_dist = likelihood(params, latent_dist)

    # Check that the distribution is a Bernoulli.
    assert isinstance(pred_dist, dx.Bernoulli)

    # Check predictive mean and variance.

    p = inv_probit(latent_mean / jnp.sqrt(1.0 + jnp.diagonal(latent_cov)))
    assert (pred_dist.mean() == p).all()
    assert (pred_dist.variance() == p * (1.0 - p)).all()


@pytest.mark.parametrize("lik", [Gaussian, Bernoulli])
@pytest.mark.parametrize("n", [1, 2, 10])
def test_conjugacy(lik: AbstractLikelihood, n: int) -> None:
    likelihood = lik(num_datapoints=n)

    # Gaussian likelihood is conjugate.
    if isinstance(likelihood, Gaussian):
        assert isinstance(likelihood, Conjugate)

    # Bernoulli likelihood is non-conjugate.
    elif isinstance(likelihood, Bernoulli):
        assert isinstance(likelihood, NonConjugate)
