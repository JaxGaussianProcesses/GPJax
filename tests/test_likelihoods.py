import jax.numpy as jnp
import jax.random as jr
import pytest

from gpjax.likelihoods import (
    Bernoulli,
    Gaussian,
    Poisson,
    initialise,
    predictive_moments,
)

true_initialisation = {"Gaussian": ["obs_noise"], "Bernoulli": [], "Poisson": []}


@pytest.mark.parametrize("lik", [Gaussian, Bernoulli, Poisson])
def test_initialisers(lik):
    l = lik()
    params = initialise(l)
    assert list(params.keys()) == true_initialisation[l.name]
    assert len(list(params.values())) == len(true_initialisation[l.name])


@pytest.mark.parametrize("n", [1, 10])
def test_predictive_moment(n):
    l = Bernoulli()
    key = jr.PRNGKey(123)
    fmean = jr.uniform(key=key, shape=(n,)) * -1
    fvar = jr.uniform(key=key, shape=(n,))
    pred_mom_fn = predictive_moments(l)
    rv = pred_mom_fn(fmean, fvar)
    mu = rv.mean()
    sigma = rv.variance()
    assert mu.shape == (n,)
    assert sigma.shape == (n,)
