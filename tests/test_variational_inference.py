import typing as tp

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import gpjax as gpx


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
@pytest.mark.parametrize("n_test", [1, 10])
@pytest.mark.parametrize("whiten", [True, False])
@pytest.mark.parametrize("diag", [True, False])
@pytest.mark.parametrize("jit_fns", [False, True])
@pytest.mark.parametrize("point_dim", [1, 2])
def test_stochastic_vi(
    n_datapoints, n_inducing_points, n_test, whiten, diag, jit_fns, point_dim
):
    D, post, prior = get_data_and_gp(n_datapoints, point_dim)
    inducing_inputs = jnp.linspace(-5.0, 5.0, n_inducing_points).reshape(-1, 1)
    inducing_inputs = jnp.hstack([inducing_inputs] * point_dim)

    if whiten is True:
        q = gpx.WhitenedVariationalGaussian(
            prior=prior, inducing_inputs=inducing_inputs, diag=diag
        )
    else:
        q = gpx.VariationalGaussian(
            prior=prior, inducing_inputs=inducing_inputs, diag=diag
        )

    svgp = gpx.StochasticVI(posterior=post, variational_family=q)
    assert svgp.posterior.prior == post.prior
    assert svgp.posterior.likelihood == post.likelihood

    params, _, _ = gpx.initialise(svgp, jr.PRNGKey(123)).unpack()

    assert svgp.prior == post.prior
    assert svgp.likelihood == post.likelihood
    assert svgp.num_inducing == n_inducing_points

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

    q = gpx.variational_families.CollapsedVariationalGaussian(
        prior=prior, likelihood=likelihood, inducing_inputs=inducing_inputs
    )

    sgpr = gpx.variational_inference.CollapsedVI(posterior=post, variational_family=q)
    assert sgpr.posterior.prior == post.prior
    assert sgpr.posterior.likelihood == post.likelihood

    params, _, _ = gpx.initialise(sgpr, jr.PRNGKey(123)).unpack()

    assert sgpr.prior == post.prior
    assert sgpr.likelihood == post.likelihood
    assert sgpr.num_inducing == n_inducing_points

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
