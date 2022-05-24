import typing as tp

import distrax as dx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

from gpjax.likelihoods import AbstractLikelihood, Bernoulli, Gaussian, Conjugate, NonConjugate
from gpjax.parameters import initialise

true_initialisation = {
    "Gaussian": ["obs_noise"],
    "Bernoulli": [],
}


@pytest.mark.parametrize("num_datapoints", [1, 10])
@pytest.mark.parametrize("lik", [Gaussian, Bernoulli])
def test_initialisers(num_datapoints, lik):
    lhood = lik(num_datapoints=num_datapoints)
    params, _, _, _ = initialise(lhood)
    assert list(params.keys()) == true_initialisation[lhood.name]
    assert len(list(params.values())) == len(true_initialisation[lhood.name])


@pytest.mark.parametrize("n", [1, 10])
def test_predictive_moment(n):
    lhood = Bernoulli(num_datapoints=n)
    key = jr.PRNGKey(123)
    fmean = jr.uniform(key=key, shape=(n,)) * -1
    fvar = jr.uniform(key=key, shape=(n,))
    pred_mom_fn = lhood.predictive_moment_fn
    rv = pred_mom_fn(fmean, fvar)
    mu = rv.mean()
    sigma = rv.variance()
    assert isinstance(lhood.predictive_moment_fn, tp.Callable)
    assert mu.shape == (n,)
    assert sigma.shape == (n,)


@pytest.mark.parametrize("lik", [Gaussian, Bernoulli])
@pytest.mark.parametrize("n", [1, 10])
def test_link_fns(lik: AbstractLikelihood, n: int):
    lhood = lik(num_datapoints=n)
    params, _, _, _ = initialise(lhood)
    link_fn = lhood.link_function
    assert isinstance(link_fn, tp.Callable)
    x = jnp.linspace(-3.0, 3.0).reshape(-1, 1)
    l_eval = link_fn(x, params)

    assert isinstance(l_eval, dx.Distribution)


@pytest.mark.parametrize("noise", [0.1, 0.5, 1.0])
def test_call(noise):
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


@pytest.mark.parametrize("lik", [Gaussian, Bernoulli])
def test_conjugacy(lik):
    likelihood = lik(num_datapoints=10)
    if isinstance(likelihood, Gaussian):
        assert isinstance(likelihood, Conjugate)
    elif isinstance(likelihood, Bernoulli):
        assert isinstance(likelihood, NonConjugate)
