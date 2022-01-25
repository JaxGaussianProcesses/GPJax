import typing as tp
from turtle import pos

import jax.numpy as jnp
import jax.random as jr
import pytest
from numpy import isin

from gpjax import Dataset, initialise, likelihoods, transform
from gpjax.gps import (
    GP,
    ConjugatePosterior,
    NonConjugatePosterior,
    Prior,
    construct_posterior,
)
from gpjax.kernels import RBF
from gpjax.likelihoods import Bernoulli, Gaussian, NonConjugateLikelihoods
from gpjax.parameters import initialise


@pytest.mark.parametrize("num_datapoints", [1, 10])
def test_prior(num_datapoints):
    p = Prior(kernel=RBF())
    params, _, _ = initialise(p)
    assert isinstance(p, Prior)
    assert isinstance(p, GP)
    meanf = p.mean(params)
    varf = p.variance(params)
    assert isinstance(meanf, tp.Callable)
    assert isinstance(varf, tp.Callable)
    x = jnp.linspace(-3.0, 3.0, num_datapoints).reshape(-1, 1)
    mu = meanf(x)
    sigma = varf(x)
    assert mu.shape == (num_datapoints, 1)
    assert sigma.shape == (num_datapoints, num_datapoints)


@pytest.mark.parametrize("num_datapoints", [2, 10])
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
    assert isinstance(post, GP)
    assert isinstance(p, GP)
    params, constrainers, unconstrainer = initialise(post)
    params = transform(params, unconstrainer)

    # Marginal likelihood
    mll = post.marginal_log_likelihood(training=D, transformations=constrainers)
    objective_val = mll(params)
    assert isinstance(objective_val, jnp.DeviceArray)
    assert objective_val.shape == ()

    # Prediction
    meanf = post.mean(training_data=D, params=params)
    varf = post.variance(training_data=D, params=params)
    assert isinstance(meanf, tp.Callable)
    assert isinstance(varf, tp.Callable)
    x = jnp.linspace(-3.0, 3.0, num_datapoints).reshape(-1, 1)
    mu = meanf(x)
    sigma = varf(x)
    assert mu.shape == (num_datapoints, 1)
    assert sigma.shape == (num_datapoints, num_datapoints)


@pytest.mark.parametrize("num_datapoints", [2, 10])
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
    assert isinstance(post, GP)
    assert isinstance(p, GP)
    params, constrainers, unconstrainer = initialise(post)
    params = transform(params, unconstrainer)

    # Marginal likelihood
    mll = post.marginal_log_likelihood(training=D, transformations=constrainers)
    objective_val = mll(params)
    assert isinstance(objective_val, jnp.DeviceArray)
    assert objective_val.shape == ()

    # Prediction
    meanf = post.mean(training_data=D, params=params)
    varf = post.variance(training_data=D, params=params)
    assert isinstance(meanf, tp.Callable)
    assert isinstance(varf, tp.Callable)
    x = jnp.linspace(-3.0, 3.0, num_datapoints).reshape(-1, 1)
    mu = meanf(x)
    sigma = varf(x)
    assert mu.shape == (num_datapoints, 1)
    assert sigma.shape == (num_datapoints, num_datapoints)


@pytest.mark.parametrize("num_datapoints", [1, 10])
@pytest.mark.parametrize("lik", [Bernoulli, Gaussian])
def test_param_construction(num_datapoints, lik):
    p = Prior(kernel=RBF()) * lik(num_datapoints=num_datapoints)
    params, _, _ = initialise(p)
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
def test_posterior_construct(lik):
    pr = Prior(kernel=RBF())
    l = lik(num_datapoints=10)
    p1 = pr * l
    p2 = construct_posterior(prior=pr, likelihood=l)
    assert type(p1) == type(p2)
