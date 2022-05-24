import typing as tp

import distrax as dx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import gpjax as gpx


def get_data_and_gp(n_datapoints):
    x = jnp.linspace(-5.0, 5.0, n_datapoints).reshape(-1, 1)
    y = jnp.sin(x) + jr.normal(key=jr.PRNGKey(123), shape=x.shape) * 0.1
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
def test_svgp(n_datapoints, n_inducing_points, n_test, whiten, diag, jit_fns):
    D, post, prior = get_data_and_gp(n_datapoints)
    inducing_inputs = jnp.linspace(-5.0, 5.0, n_inducing_points).reshape(-1, 1)
    test_inputs = jnp.linspace(-5.0, 5.0, n_test).reshape(-1, 1)

    if whiten is True:
        q = gpx.WhitenedVariationalGaussian(prior = prior,
        inducing_inputs=inducing_inputs, diag=diag
        )
    else:
        q = gpx.VariationalGaussian(prior = prior,
        inducing_inputs=inducing_inputs, diag=diag
        )

    svgp = gpx.SVGP(posterior=post, variational_family=q)

    assert svgp.posterior.prior == post.prior
    assert svgp.posterior.likelihood == post.likelihood

    params, _, constrainer, unconstrainer = gpx.initialise(svgp)
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
    kl_q_p = q.prior_kl(constrained_params)
    assert isinstance(kl_q_p, jnp.ndarray)

    # Test predictions
    predictive_dist_fn = q(constrained_params)
    assert isinstance(predictive_dist_fn, tp.Callable)

    predictive_dist = predictive_dist_fn(test_inputs)
    assert isinstance(predictive_dist, dx.Distribution)

    mu = predictive_dist.mean()
    sigma = predictive_dist.covariance()

    assert isinstance(mu, jnp.ndarray)
    assert isinstance(sigma, jnp.ndarray)
    assert mu.shape == (n_test,)
    assert sigma.shape == (n_test, n_test)
