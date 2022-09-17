import typing as tp

import distrax as dx
import jax.numpy as jnp
import jax.random as jr
import pytest

from gpjax.likelihoods import (
    AbstractLikelihood,
    Bernoulli,
    Conjugate,
    Gaussian,
    NonConjugate,
)
from gpjax.parameters import initialise

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
def test_call_gaussian(noise):
    key = jr.PRNGKey(123)
    n = 10
    lhood = Gaussian(num_datapoints=n)
    dist = dx.MultivariateNormalFullCovariance(jnp.zeros(n), jnp.eye(n))
    params = {"likelihood": {"obs_noise": noise}}

    l_dist = lhood(dist, params)
    assert (l_dist.mean() == jnp.zeros(n)).all()
    noise_mat = jnp.diag(jnp.repeat(noise, n))
    assert (l_dist.covariance() == jnp.eye(n) + noise_mat).all()

    l_dist = lhood.predict(dist, params)
    assert (l_dist.mean() == jnp.zeros(n)).all()
    noise_mat = jnp.diag(jnp.repeat(noise, n))
    assert (l_dist.covariance() == jnp.eye(n) + noise_mat).all()


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
