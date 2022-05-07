import typing as tp

import jax
import jax.numpy as jnp
import jax.random as jr
import py
import pytest

import gpjax as gpx


def get_data_and_gp(n_datapoints):
    x = jnp.linspace(-5.0, 5.0, n_datapoints).reshape(-1, 1)
    y = jnp.sin(x) + jr.normal(key=jr.PRNGKey(123), shape=x.shape) * 0.1
    D = gpx.Dataset(X=x, y=y)

    p = gpx.Prior(kernel=gpx.RBF())
    lik = gpx.Gaussian(num_datapoints=n_datapoints)
    post = p * lik
    return D, post


@pytest.mark.parametrize("n_datapoints, n_inducing_points", [(10, 2), (100, 10)])
@pytest.mark.parametrize("n_test", [1, 10])
@pytest.mark.parametrize("whiten", [True, False])
@pytest.mark.parametrize("diag", [True, False])
@pytest.mark.parametrize("jit_fns", [True, False])
def test_svgp(n_datapoints, n_inducing_points, n_test, whiten, diag, jit_fns):
    D, post = get_data_and_gp(n_datapoints)

    inducing_inputs = jnp.linspace(-5.0, 5.0, n_inducing_points).reshape(-1, 1)
    test_inputs = jnp.linspace(-5.0, 5.0, n_test).reshape(-1, 1)
    q = gpx.VariationalGaussian(
        inducing_inputs=inducing_inputs, whiten=whiten, diag=diag
    )
    svgp = gpx.SVGP(posterior=post, variational_family=q)

    assert svgp.posterior.prior == post.prior
    assert svgp.posterior.likelihood == post.likelihood

    params, trainable_status, constrainer, unconstrainer = gpx.initialise(svgp)
    params = gpx.transform(params, unconstrainer)

    assert svgp.prior == post.prior
    assert svgp.likelihood == post.likelihood
    assert svgp.num_inducing == n_inducing_points

    if jit_fns:
        elbo_fn = jax.jit(svgp.elbo(D, constrainer))
    else:
        elbo_fn = svgp.elbo(D, constrainer)
    assert isinstance(elbo_fn, tp.Callable)
    elbo_value = elbo_fn(params, D)
    assert isinstance(elbo_value, jnp.ndarray)

    # Test gradients
    grads = jax.grad(elbo_fn, argnums=0)(params, D)
    assert isinstance(grads, tp.Dict)
    assert len(grads) == len(params)

    constrained_params = gpx.transform(params, constrainer)
    kl_q_p = svgp.prior_kl(constrained_params)
    assert isinstance(kl_q_p, jnp.ndarray)

    latent_mean, latent_cov = svgp.pred_moments(constrained_params, test_inputs)
    assert isinstance(latent_mean, jnp.ndarray)
    assert isinstance(latent_cov, jnp.ndarray)
    assert latent_mean.shape == (n_test, 1)
    assert latent_cov.shape == (n_test, n_test)

    # Test predictions
    pred_mean_fn = svgp.mean(constrained_params)
    pred_var_fn = svgp.variance(constrained_params)
    mu = pred_mean_fn(test_inputs)
    sigma = pred_var_fn(test_inputs)

    assert isinstance(pred_mean_fn, tp.Callable)
    assert isinstance(pred_var_fn, tp.Callable)
    assert isinstance(mu, jnp.ndarray)
    assert isinstance(sigma, jnp.ndarray)
    assert mu.shape == (n_test, 1)
    assert sigma.shape == (n_test, n_test)
