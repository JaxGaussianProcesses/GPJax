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
import jax.numpy as jnp
import jax.random as jr
import pytest
from jax.config import config

import gpjax as gpx
from gpjax.variational_families import (
    CollapsedVariationalGaussian,
    ExpectationVariationalGaussian,
    NaturalVariationalGaussian,
    VariationalGaussian,
    WhitenedVariationalGaussian,
)

# Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)


def test_abstract_variational_inference():
    prior = gpx.Prior(kernel=gpx.RBF())
    lik = gpx.Gaussian(num_datapoints=20)
    post = prior * lik
    n_inducing_points = 10
    inducing_inputs = jnp.linspace(-5.0, 5.0, n_inducing_points).reshape(-1, 1)
    vartiational_family = gpx.VariationalGaussian(
        prior=prior, inducing_inputs=inducing_inputs
    )

    with pytest.raises(TypeError):
        gpx.variational_inference.AbstractVariationalInference(
            posterior=post, vartiational_family=vartiational_family
        )


def get_data_and_gp(n_datapoints, point_dim):
    x = jnp.linspace(-5.0, 5.0, n_datapoints).reshape(-1, 1)
    y = jnp.sin(x) + jr.normal(key=jr.PRNGKey(123), shape=x.shape) * 0.1
    x = jnp.hstack([x] * point_dim)
    D = gpx.Dataset(X=x, y=y)

    p = gpx.Prior(kernel=gpx.RBF())
    lik = gpx.Gaussian(num_datapoints=n_datapoints)
    post = p * lik
    return D, post, p


@pytest.mark.parametrize("n_datapoints, n_inducing_points", [(10, 2), (100, 10)])
@pytest.mark.parametrize("jit_fns", [False, True])
@pytest.mark.parametrize("point_dim", [1, 2, 3])
@pytest.mark.parametrize(
    "variational_family",
    [
        VariationalGaussian,
        WhitenedVariationalGaussian,
        NaturalVariationalGaussian,
        ExpectationVariationalGaussian,
    ],
)
def test_stochastic_vi(
    n_datapoints, n_inducing_points, jit_fns, point_dim, variational_family
):
    D, post, prior = get_data_and_gp(n_datapoints, point_dim)
    inducing_inputs = jnp.linspace(-5.0, 5.0, n_inducing_points).reshape(-1, 1)
    inducing_inputs = jnp.hstack([inducing_inputs] * point_dim)

    q = variational_family(prior=prior, inducing_inputs=inducing_inputs)

    svgp = gpx.StochasticVI(posterior=post, variational_family=q)
    assert svgp.posterior.prior == post.prior
    assert svgp.posterior.likelihood == post.likelihood

    params, _, _ = gpx.initialise(svgp, jr.PRNGKey(123)).unpack()

    assert svgp.prior == post.prior
    assert svgp.likelihood == post.likelihood

    if jit_fns:
        elbo_fn = jax.jit(svgp.elbo(D))
    else:
        elbo_fn = svgp.elbo(D)
    assert isinstance(elbo_fn, tp.Callable)
    elbo_value = elbo_fn(params, D)
    assert isinstance(elbo_value, jnp.ndarray)

    # Test gradients
    grads = jax.grad(elbo_fn, argnums=0)(params, D)
    assert isinstance(grads, tp.Dict)
    assert len(grads) == len(params)


@pytest.mark.parametrize("n_datapoints, n_inducing_points", [(10, 2), (100, 10)])
@pytest.mark.parametrize("jit_fns", [False, True])
@pytest.mark.parametrize("point_dim", [1, 2])
def test_collapsed_vi(n_datapoints, n_inducing_points, jit_fns, point_dim):
    D, post, prior = get_data_and_gp(n_datapoints, point_dim)
    likelihood = gpx.Gaussian(num_datapoints=n_datapoints)

    inducing_inputs = jnp.linspace(-5.0, 5.0, n_inducing_points).reshape(-1, 1)
    inducing_inputs = jnp.hstack([inducing_inputs] * point_dim)

    q = CollapsedVariationalGaussian(
        prior=prior, likelihood=likelihood, inducing_inputs=inducing_inputs
    )

    sgpr = gpx.variational_inference.CollapsedVI(posterior=post, variational_family=q)
    assert sgpr.posterior.prior == post.prior
    assert sgpr.posterior.likelihood == post.likelihood

    params, _, _ = gpx.initialise(sgpr, jr.PRNGKey(123)).unpack()

    assert sgpr.prior == post.prior
    assert sgpr.likelihood == post.likelihood

    if jit_fns:
        elbo_fn = jax.jit(sgpr.elbo(D))
    else:
        elbo_fn = sgpr.elbo(D)
    assert isinstance(elbo_fn, tp.Callable)
    elbo_value = elbo_fn(params)
    assert isinstance(elbo_value, jnp.ndarray)

    # Test gradients
    grads = jax.grad(elbo_fn)(params)
    assert isinstance(grads, tp.Dict)
    assert len(grads) == len(params)

    # We should raise an error for non-Collapsed variational families:
    with pytest.raises(TypeError):
        q = gpx.variational_families.VariationalGaussian(
            prior=prior, inducing_inputs=inducing_inputs
        )
        gpx.variational_inference.CollapsedVI(posterior=post, variational_family=q)

    # We should raise an error for non-Gaussian likelihoods:
    with pytest.raises(TypeError):
        q = gpx.variational_families.CollapsedVariationalGaussian(
            prior=prior, likelihood=likelihood, inducing_inputs=inducing_inputs
        )
        gpx.variational_inference.CollapsedVI(
            posterior=prior * gpx.Bernoulli(num_datapoints=D.n), variational_family=q
        )
