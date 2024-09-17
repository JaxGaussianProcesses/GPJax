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

try:
    from beartype.roar import BeartypeCallHintParamViolation

    ValidationErrors = (TypeError, BeartypeCallHintParamViolation)
except ImportError:
    ValidationErrors = TypeError

from typing import (
    Callable,
    Type,
)

from cola.linalg.algorithm_base import Auto
from cola.linalg.decompositions.decompositions import Cholesky
from cola.linalg.inverse.cg import CG
from jax import config
import jax.numpy as jnp
import jax.random as jr
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

from gpjax.dataset import Dataset
from gpjax.distributions import GaussianDistribution
from gpjax.gps import (
    AbstractPosterior,
    AbstractPrior,
    ConjugatePosterior,
    NonConjugatePosterior,
    Prior,
    construct_posterior,
)
from gpjax.kernels import (
    RBF,
    AbstractKernel,
    Matern52,
)
from gpjax.likelihoods import (
    AbstractLikelihood,
    Bernoulli,
    Gaussian,
    Poisson,
)
from gpjax.mean_functions import (
    AbstractMeanFunction,
    Constant,
    Zero,
)

# Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)


def test_abstract_prior():
    # Abstract prior should not be able to be instantiated.
    with pytest.raises(TypeError):
        AbstractPrior()


def test_abstract_posterior():
    # Abstract posterior should not be able to be instantiated.
    with pytest.raises(TypeError):
        AbstractPosterior()


@pytest.mark.parametrize("num_datapoints", [1, 10])
@pytest.mark.parametrize("kernel", [RBF, Matern52])
@pytest.mark.parametrize("mean_function", [Zero, Constant])
def test_prior(
    num_datapoints: int,
    kernel: type[AbstractKernel],
    mean_function: Type[AbstractMeanFunction],
) -> None:
    # Create prior.
    prior = Prior(mean_function=mean_function(), kernel=kernel())

    # Check types.
    assert isinstance(prior, Prior)
    assert isinstance(prior, AbstractPrior)

    # Query a marginal distribution at some inputs.
    inputs = jnp.linspace(-3.0, 3.0, num_datapoints).reshape(-1, 1)
    marginal_distribution = prior(inputs)

    # Ensure that the marginal distribution is a Gaussian.
    assert isinstance(marginal_distribution, GaussianDistribution)
    assert isinstance(marginal_distribution, tfd.Distribution)

    # Ensure that the marginal distribution has the correct shape.
    mu = marginal_distribution.mean()
    sigma = marginal_distribution.covariance()
    assert mu.shape == (num_datapoints,)
    assert sigma.shape == (num_datapoints, num_datapoints)


@pytest.mark.parametrize("num_datapoints", [1, 10])
@pytest.mark.parametrize("kernel", [RBF, Matern52])
@pytest.mark.parametrize("mean_function", [Zero, Constant])
def test_conjugate_posterior(
    num_datapoints: int,
    kernel: type[AbstractKernel],
    mean_function: type[AbstractMeanFunction],
) -> None:
    # Create a dataset.
    key = jr.key(123)
    x = jr.uniform(key=key, minval=-2.0, maxval=2.0, shape=(num_datapoints, 1))
    y = jnp.sin(x) + jr.normal(key=key, shape=x.shape) * 0.1
    D = Dataset(X=x, y=y)

    # Define prior.
    prior = Prior(mean_function=mean_function(), kernel=kernel())

    # Define a likelihood.
    likelihood = Gaussian(num_datapoints=num_datapoints)

    # Construct the posterior via the class.
    posterior = ConjugatePosterior(prior=prior, likelihood=likelihood)

    # Check types.
    assert isinstance(posterior, ConjugatePosterior)

    # Query a marginal distribution of the posterior at some inputs.
    inputs = jnp.linspace(-3.0, 3.0, num_datapoints).reshape(-1, 1)
    marginal_distribution = posterior(inputs, D)

    # Ensure that the marginal distribution is a Gaussian.
    assert isinstance(marginal_distribution, GaussianDistribution)
    assert isinstance(marginal_distribution, tfd.Distribution)

    # Ensure that the marginal distribution has the correct shape.
    mu = marginal_distribution.mean()
    sigma = marginal_distribution.covariance()
    assert mu.shape == (num_datapoints,)
    assert sigma.shape == (num_datapoints, num_datapoints)


@pytest.mark.parametrize("num_datapoints", [1, 10])
@pytest.mark.parametrize("kernel", [RBF, Matern52])
@pytest.mark.parametrize("mean_function", [Zero, Constant])
def test_nonconjugate_posterior(
    num_datapoints: int,
    kernel: type[AbstractKernel],
    mean_function: type[AbstractMeanFunction],
) -> None:
    # Create a dataset.
    key = jr.key(123)
    x = jr.uniform(key=key, minval=-2.0, maxval=2.0, shape=(num_datapoints, 1))
    y = jnp.sin(x) + jr.normal(key=key, shape=x.shape) * 0.1
    D = Dataset(X=x, y=y)

    # Define prior.
    prior = Prior(mean_function=mean_function(), kernel=kernel())

    # Define a likelihood.
    likelihood = Bernoulli(num_datapoints=num_datapoints)

    # Construct the posterior via the class.
    posterior = NonConjugatePosterior(prior=prior, likelihood=likelihood)

    # Check types.
    assert isinstance(posterior, NonConjugatePosterior)

    # Check latent values.
    latent_values = jr.normal(posterior.key.value, (num_datapoints, 1))
    assert (posterior.latent.value == latent_values).all()

    # Query a marginal distribution of the posterior at some inputs.
    inputs = jnp.linspace(-3.0, 3.0, num_datapoints).reshape(-1, 1)
    marginal_distribution = posterior(inputs, D)

    # Ensure that the marginal distribution is a Gaussian.
    assert isinstance(marginal_distribution, GaussianDistribution)
    assert isinstance(marginal_distribution, tfd.Distribution)

    # Ensure that the marginal distribution has the correct shape.
    mu = marginal_distribution.mean()
    sigma = marginal_distribution.covariance()
    assert mu.shape == (num_datapoints,)
    assert sigma.shape == (num_datapoints, num_datapoints)


@pytest.mark.parametrize("likelihood", [Bernoulli, Gaussian])
@pytest.mark.parametrize("num_datapoints", [1, 10])
@pytest.mark.parametrize("kernel", [RBF, Matern52])
@pytest.mark.parametrize("mean_function", [Zero, Constant])
def test_posterior_construct(
    likelihood: Type[AbstractLikelihood],
    num_datapoints: int,
    kernel: type[AbstractKernel],
    mean_function: type[AbstractMeanFunction],
) -> None:
    # Define prior.
    prior = Prior(mean_function=mean_function(), kernel=kernel())

    # Construct the posterior via the three methods.
    likelihood: AbstractLikelihood = likelihood(num_datapoints=num_datapoints)
    posterior_mul = prior * likelihood
    posterior_rmul = likelihood * prior
    posterior_manual = construct_posterior(prior=prior, likelihood=likelihood)

    # Ensure that the posterior is the same type in all three cases.
    assert type(posterior_mul) is type(posterior_rmul)
    assert type(posterior_mul) is type(posterior_manual)

    # Ensure we have the correct likelihood and prior.
    assert posterior_mul.likelihood == likelihood
    assert posterior_mul.prior == prior

    # If the likelihood is Gaussian, then the posterior should be conjugate.
    if isinstance(likelihood, Gaussian):
        assert isinstance(posterior_mul, ConjugatePosterior)

    # If the likelihood is Bernoulli or Poisson, then the posterior should be non-conjugate.
    if isinstance(likelihood, (Bernoulli, Poisson)):
        assert isinstance(posterior_mul, NonConjugatePosterior)


@pytest.mark.parametrize("num_datapoints", [1, 5])
@pytest.mark.parametrize("kernel", [RBF, Matern52])
@pytest.mark.parametrize("mean_function", [Zero, Constant])
def test_prior_sample_approx(num_datapoints, kernel, mean_function):
    kern = kernel(n_dims=2, lengthscale=jnp.array([5.0, 1.0]), variance=0.1)
    p = Prior(kernel=kern, mean_function=mean_function())
    key = jr.PRNGKey(123)

    with pytest.raises(ValueError):
        p.sample_approx(-1, key)
    with pytest.raises(ValueError):
        p.sample_approx(0, key)
    with pytest.raises(ValidationErrors):
        p.sample_approx(0.5, key)
    with pytest.raises(ValueError):
        p.sample_approx(1, key, -10)
    with pytest.raises(ValueError):
        p.sample_approx(1, key, 0)
    with pytest.raises(ValidationErrors):
        p.sample_approx(1, key, 0.5)

    sampled_fn = p.sample_approx(1, key, 100)
    assert isinstance(sampled_fn, Callable)  # check type

    x = jr.uniform(key=key, minval=-2.0, maxval=2.0, shape=(num_datapoints, 2))
    evals = sampled_fn(x)
    assert evals.shape == (num_datapoints, 1.0)  # check shape

    sampled_fn_2 = p.sample_approx(1, key, 100)
    evals_2 = sampled_fn_2(x)
    max_delta = jnp.max(jnp.abs(evals - evals_2))
    assert max_delta == 0.0  # samples same for same seed

    new_key = jr.key(12345)
    sampled_fn_3 = p.sample_approx(1, new_key, 100)
    evals_3 = sampled_fn_3(x)
    max_delta = jnp.max(jnp.abs(evals - evals_3))
    assert max_delta > 0.01  # samples different for different seed

    # Check validty of samples using Monte-Carlo
    sampled_fn = p.sample_approx(10_000, key, 100)
    sampled_evals = sampled_fn(x)
    approx_mean = jnp.mean(sampled_evals, -1)
    approx_var = jnp.var(sampled_evals, -1)
    true_predictive = p(x)
    true_mean = true_predictive.mean()
    true_var = jnp.diagonal(true_predictive.covariance())
    max_error_in_mean = jnp.max(jnp.abs(approx_mean - true_mean))
    max_error_in_var = jnp.max(jnp.abs(approx_var - true_var))
    assert max_error_in_mean < 0.02  # check that samples are correct
    assert max_error_in_var < 0.05  # check that samples are correct


@pytest.mark.parametrize("num_datapoints", [1, 5])
@pytest.mark.parametrize("kernel", [RBF, Matern52])
@pytest.mark.parametrize("mean_function", [Zero, Constant])
@pytest.mark.parametrize("solver_algorithm", [Cholesky(), CG(), Auto()])
def test_conjugate_posterior_sample_approx(
    num_datapoints, kernel, mean_function, solver_algorithm
):
    kern = kernel(lengthscale=jnp.array([5.0, 1.0]), variance=0.1)
    p = Prior(kernel=kern, mean_function=mean_function()) * Gaussian(
        num_datapoints=num_datapoints
    )
    key = jr.key(123)

    x = jr.uniform(key=key, minval=-2.0, maxval=2.0, shape=(num_datapoints, 2))
    y = (
        jnp.mean(jnp.sin(x), 1, keepdims=True)
        + jr.normal(key=key, shape=(num_datapoints, 1)) * 0.1
    )
    D = Dataset(X=x, y=y)

    # with pytest.raises(ValueError):
    #     p.sample_approx(-1, D, key)
    # with pytest.raises(ValueError):
    #     p.sample_approx(0, D, key)
    # with pytest.raises(ValidationErrors):
    #     p.sample_approx(0.5, D, key)
    # with pytest.raises(ValueError):
    #     p.sample_approx(1, D, key, -10)
    # with pytest.raises(ValueError):
    #     p.sample_approx(1, D, key, 0)
    # with pytest.raises(ValidationErrors):
    #     p.sample_approx(1, D, key, 0.5)

    sampled_fn = p.sample_approx(1, D, key, 100, solver_algorithm=solver_algorithm)
    assert isinstance(sampled_fn, Callable)  # check type

    x = jr.uniform(key=key, minval=-2.0, maxval=2.0, shape=(num_datapoints, 2))
    evals = sampled_fn(x)
    assert evals.shape == (num_datapoints, 1.0)  # check shape

    sampled_fn_2 = p.sample_approx(1, D, key, 100, solver_algorithm=solver_algorithm)
    evals_2 = sampled_fn_2(x)
    max_delta = jnp.max(jnp.abs(evals - evals_2))
    assert max_delta == 0.0  # samples same for same seed

    new_key = jr.key(12345)
    sampled_fn_3 = p.sample_approx(
        1, D, new_key, 100, solver_algorithm=solver_algorithm
    )
    evals_3 = sampled_fn_3(x)
    max_delta = jnp.max(jnp.abs(evals - evals_3))
    assert max_delta > 0.01  # samples different for different seed

    # Check validty of samples using Monte-Carlo
    sampled_fn = p.sample_approx(10_000, D, key, 100, solver_algorithm=solver_algorithm)
    sampled_evals = sampled_fn(x)
    approx_mean = jnp.mean(sampled_evals, -1)
    approx_var = jnp.var(sampled_evals, -1)
    true_predictive = p(x, train_data=D)
    true_mean = true_predictive.mean()
    true_var = jnp.diagonal(true_predictive.covariance())
    max_error_in_mean = jnp.max(jnp.abs(approx_mean - true_mean))
    max_error_in_var = jnp.max(jnp.abs(approx_var - true_var))
    assert max_error_in_mean < 0.02  # check that samples are correct
    assert max_error_in_var < 0.05  # check that samples are correct


if __name__ == "__main__":
    test_conjugate_posterior_sample_approx(10, RBF, Zero)
