import jax.numpy as jnp
import jax.random as jr
import pytest

from gpjax.likelihoods import (
    Likelihood,
    Bernoulli,
    Gaussian,
)
from gpjax.parameters import initialise
import typing as tp

true_initialisation = {
    "Gaussian": ["obs_noise"],
    "Bernoulli": [],
}


@pytest.mark.parametrize("num_datapoints", [1, 10])
@pytest.mark.parametrize("lik", [Gaussian, Bernoulli])
def test_initialisers(num_datapoints, lik):
    l = lik(num_datapoints=num_datapoints)
    params, _, _ = initialise(l)
    assert list(params.keys()) == true_initialisation[l.name]
    assert len(list(params.values())) == len(true_initialisation[l.name])


@pytest.mark.parametrize("n", [1, 10])
def test_predictive_moment(n):
    l = Bernoulli(num_datapoints=n)
    key = jr.PRNGKey(123)
    fmean = jr.uniform(key=key, shape=(n,)) * -1
    fvar = jr.uniform(key=key, shape=(n,))
    pred_mom_fn = l.predictive_moment_fn
    rv = pred_mom_fn(fmean, fvar)
    mu = rv.mean()
    sigma = rv.variance()
    assert isinstance(l.predictive_moment_fn, tp.Callable)
    assert mu.shape == (n,)
    assert sigma.shape == (n,)


@pytest.mark.parametrize("lik", [Gaussian, Bernoulli])
@pytest.mark.parametrize("n", [1, 10])
def test_link_fns(lik: Likelihood, n: int):
    l = lik(num_datapoints=n)
    assert isinstance(l.link_function, tp.Callable)
