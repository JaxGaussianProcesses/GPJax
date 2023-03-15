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

import typing as tp

import jax
import distrax as dx
import jax.numpy as jnp
import jax.random as jr
import pytest
from jax.config import config
from jaxutils import Dataset, Parameters


from gpjax.gps import (
    AbstractPrior,
    AbstractPosterior,
    ConjugatePosterior,
    NonConjugatePosterior,
    Prior,
    construct_posterior,
)
from jaxkern import RBF
from gpjax.likelihoods import Bernoulli, Gaussian

# Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)
NonConjugateLikelihoods = [Bernoulli]


@pytest.mark.parametrize("num_datapoints", [1, 10])
def test_prior(num_datapoints):
    p = Prior(kernel=RBF())
    parameters = p.init_params(jr.PRNGKey(123))

    assert isinstance(p, Prior)
    assert isinstance(p, AbstractPrior)
    assert isinstance(parameters, Parameters)

    prior_rv_fn = p(parameters)
    assert isinstance(prior_rv_fn, tp.Callable)

    x = jnp.linspace(-3.0, 3.0, num_datapoints).reshape(-1, 1)
    predictive_dist = prior_rv_fn(x)
    assert isinstance(predictive_dist, dx.Distribution)
    mu = predictive_dist.mean()
    sigma = predictive_dist.covariance()
    assert mu.shape == (num_datapoints,)
    assert sigma.shape == (num_datapoints, num_datapoints)


@pytest.mark.parametrize("num_datapoints", [1, 2, 10])
def test_conjugate_posterior(num_datapoints):
    key = jr.PRNGKey(123)
    x = jnp.sort(
        jr.uniform(key=key, minval=-2.0, maxval=2.0, shape=(num_datapoints, 1)),
        axis=0,
    )
    y = jnp.sin(x) + jr.normal(key=key, shape=x.shape) * 0.1
    D = Dataset(X=x, y=y)

    # Initialisation
    p = Prior(kernel=RBF())
    lik = Gaussian(num_datapoints=num_datapoints)
    post = p * lik
    assert isinstance(post, ConjugatePosterior)
    assert isinstance(post, AbstractPrior)
    assert isinstance(p, AbstractPrior)

    post2 = lik * p
    assert isinstance(post2, ConjugatePosterior)
    assert isinstance(post2, AbstractPrior)

    params = post.init_params(key)
    assert isinstance(params, Parameters)
    print(params.params)

    # Prediction
    predictive_dist_fn = post(params, D)
    assert isinstance(predictive_dist_fn, tp.Callable)

    x = jnp.linspace(-3.0, 3.0, num_datapoints).reshape(-1, 1)
    predictive_dist = predictive_dist_fn(x)
    assert isinstance(predictive_dist, dx.Distribution)

    mu = predictive_dist.mean()
    sigma = predictive_dist.covariance()
    assert mu.shape == (num_datapoints,)
    assert sigma.shape == (num_datapoints, num_datapoints)


@pytest.mark.parametrize("num_datapoints", [1, 2, 10])
@pytest.mark.parametrize("likel", NonConjugateLikelihoods)
def test_nonconjugate_posterior(num_datapoints, likel):
    key = jr.PRNGKey(123)
    x = jnp.sort(
        jr.uniform(key=key, minval=-2.0, maxval=2.0, shape=(num_datapoints, 1)),
        axis=0,
    )
    y = 0.5 * jnp.sign(jnp.cos(3 * x + jr.normal(key, shape=x.shape) * 0.05)) + 0.5
    D = Dataset(X=x, y=y)
    # Initialisation
    p = Prior(kernel=RBF())
    lik = likel(num_datapoints=num_datapoints)
    post = p * lik
    assert isinstance(post, NonConjugatePosterior)
    assert isinstance(post, AbstractPrior)
    assert isinstance(p, AbstractPrior)

    params = post.init_params(key)
    assert isinstance(params, Parameters)

    # Marginal likelihood
    mll = post.marginal_log_likelihood(train_data=D)
    objective_val = mll(params)
    assert isinstance(objective_val, jax.Array)
    assert objective_val.shape == ()

    # Prediction
    predictive_dist_fn = post(params, D)
    assert isinstance(predictive_dist_fn, tp.Callable)

    x = jnp.linspace(-3.0, 3.0, num_datapoints).reshape(-1, 1)
    predictive_dist = predictive_dist_fn(x)
    assert isinstance(predictive_dist, dx.Distribution)

    mu = predictive_dist.mean()
    sigma = predictive_dist.covariance()
    assert mu.shape == (num_datapoints,)
    assert sigma.shape == (num_datapoints, num_datapoints)


@pytest.mark.parametrize("num_datapoints", [1, 10])
@pytest.mark.parametrize("lik", [Bernoulli, Gaussian])
def test_param_construction(num_datapoints, lik):
    p = Prior(kernel=RBF()) * lik(num_datapoints=num_datapoints)
    params = p.init_params(jr.PRNGKey(123))

    if isinstance(lik, Bernoulli):
        assert sorted(list(params.keys())) == [
            "kernel",
            "latent_fn",
            "likelihood",
            "mean_function",
        ]
    elif isinstance(lik, Gaussian):
        assert sorted(list(params.keys())) == [
            "kernel",
            "likelihood",
            "mean_function",
        ]


@pytest.mark.parametrize("lik", [Bernoulli, Gaussian])
def test_abstract_posterior(lik):
    pr = Prior(kernel=RBF())
    likelihood = lik(num_datapoints=10)

    with pytest.raises(TypeError):
        _ = AbstractPosterior(pr, likelihood)

    class DummyPosterior(AbstractPosterior):
        def predict(self):
            pass

    dummy_post = DummyPosterior(pr, likelihood)
    assert isinstance(dummy_post, AbstractPosterior)
    assert dummy_post.likelihood == likelihood
    assert dummy_post.prior == pr


@pytest.mark.parametrize("lik", [Bernoulli, Gaussian])
def test_posterior_construct(lik):
    pr = Prior(kernel=RBF())
    likelihood = lik(num_datapoints=10)
    p1 = pr * likelihood
    p2 = construct_posterior(prior=pr, likelihood=likelihood)
    assert type(p1) == type(p2)
