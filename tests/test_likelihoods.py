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

import distrax as dx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax.config import config

from gpjax.likelihoods import (
    AbstractLikelihood,
    Bernoulli,
    Conjugate,
    Gaussian,
    NonConjugate,
)
from gpjax.parameters import initialise

# Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)
true_initialisation = {
    "Gaussian": ["obs_noise"],
    "Bernoulli": [],
}


@pytest.mark.parametrize("num_datapoints", [1, 10])
@pytest.mark.parametrize("lik", [Gaussian, Bernoulli])
def test_initialisers(num_datapoints, lik):
    key = jr.PRNGKey(123)
    lhood = lik(num_datapoints=num_datapoints)
    params, _, _ = initialise(lhood, key).unpack()
    assert list(params.keys()) == true_initialisation[lhood.name]
    assert len(list(params.values())) == len(true_initialisation[lhood.name])


@pytest.mark.parametrize("n", [1, 10])
def test_predictive_moment(n):
    lhood = Bernoulli(num_datapoints=n)
    key = jr.PRNGKey(123)
    fmean = jr.uniform(key=key, shape=(n,)) * -1
    fvar = jr.uniform(key=key, shape=(n,))
    pred_mom_fn = lhood.predictive_moment_fn
    params, _, _ = initialise(lhood, key).unpack()
    rv = pred_mom_fn(fmean, fvar, params)
    mu = rv.mean()
    sigma = rv.variance()
    assert isinstance(lhood.predictive_moment_fn, tp.Callable)
    assert mu.shape == (n,)
    assert sigma.shape == (n,)


@pytest.mark.parametrize("lik", [Gaussian, Bernoulli])
@pytest.mark.parametrize("n", [1, 10])
def test_link_fns(lik: AbstractLikelihood, n: int):
    key = jr.PRNGKey(123)
    lhood = lik(num_datapoints=n)
    params, _, _ = initialise(lhood, key).unpack()
    link_fn = lhood.link_function
    assert isinstance(link_fn, tp.Callable)
    x = jnp.linspace(-3.0, 3.0).reshape(-1, 1)
    l_eval = link_fn(x, params)

    assert isinstance(l_eval, dx.Distribution)


@pytest.mark.parametrize("noise", [0.1, 0.5, 1.0])
@pytest.mark.parametrize("n", [1, 2, 10])
def test_call_gaussian(noise, n):
    key = jr.PRNGKey(123)
    lhood = Gaussian(num_datapoints=n)
    dist = dx.MultivariateNormalFullCovariance(jnp.zeros(n), jnp.eye(n))
    params = {"likelihood": {"obs_noise": noise}}

    l_dist = lhood(dist, params)
    assert (l_dist.mean() == jnp.zeros(n)).all()
    noise_mat = jnp.diag(jnp.repeat(noise, n))
    assert np.allclose(l_dist.scale_tri, jnp.linalg.cholesky(jnp.eye(n) + noise_mat))
    l_dist = lhood.predict(dist, params)
    assert (l_dist.mean() == jnp.zeros(n)).all()
    noise_mat = jnp.diag(jnp.repeat(noise, n))
    assert np.allclose(l_dist.scale_tri, jnp.linalg.cholesky(jnp.eye(n) + noise_mat))


def test_call_bernoulli():
    n = 10
    lhood = Bernoulli(num_datapoints=n)
    dist = dx.MultivariateNormalFullCovariance(jnp.zeros(n), jnp.eye(n))
    params = {"likelihood": {}}

    l_dist = lhood(dist, params)
    assert (l_dist.mean() == 0.5 * jnp.ones(n)).all()
    assert (l_dist.variance() == 0.25 * jnp.ones(n)).all()

    l_dist = lhood.predict(dist, params)
    assert (l_dist.mean() == 0.5 * jnp.ones(n)).all()
    assert (l_dist.variance() == 0.25 * jnp.ones(n)).all()


@pytest.mark.parametrize("lik", [Gaussian, Bernoulli])
def test_conjugacy(lik):
    likelihood = lik(num_datapoints=10)
    if isinstance(likelihood, Gaussian):
        assert isinstance(likelihood, Conjugate)
    elif isinstance(likelihood, Bernoulli):
        assert isinstance(likelihood, NonConjugate)
