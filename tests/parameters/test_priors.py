import jax.numpy as jnp
import pytest
from tensorflow_probability.substrates.jax import distributions as tfd

from gpjax.gps import Prior
from gpjax.kernels import RBF
from gpjax.likelihoods import Bernoulli
from gpjax.parameters import log_density
from gpjax.parameters.priors import evaluate_prior, prior_checks


@pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
def test_lpd(x):
    val = jnp.array(x)
    dist = tfd.Normal(loc=0.0, scale=1.0)
    lpd = log_density(val, dist)
    assert lpd is not None


def test_prior_evaluation():
    """
    Test the regular setup that every parameter has a corresponding prior distribution attached to its unconstrained
    value.
    """
    params = {
        "lengthscale": jnp.array([1.0]),
        "variance": jnp.array([1.0]),
        "obs_noise": jnp.array([1.0]),
    }
    priors = {
        "lengthscale": tfd.Gamma(1.0, 1.0),
        "variance": tfd.Gamma(2.0, 2.0),
        "obs_noise": tfd.Gamma(3.0, 3.0),
    }
    lpd = evaluate_prior(params, priors)
    assert pytest.approx(lpd) == -2.0110168


def test_none_prior():
    """
    Test that multiple dispatch is working in the case of no priors.
    """
    params = {
        "lengthscale": jnp.array([1.0]),
        "variance": jnp.array([1.0]),
        "obs_noise": jnp.array([1.0]),
    }
    lpd = evaluate_prior(params, None)
    assert lpd == 0.0


def test_incomplete_priors():
    """
    Test the case where a user specifies priors for some, but not all, parameters.
    """
    params = {
        "lengthscale": jnp.array([1.0]),
        "variance": jnp.array([1.0]),
        "obs_noise": jnp.array([1.0]),
    }
    priors = {
        "lengthscale": tfd.Gamma(1.0, 1.0),
        "variance": tfd.Gamma(2.0, 2.0),
    }
    lpd = evaluate_prior(params, priors)
    assert pytest.approx(lpd) == -1.6137061


def test_checks():
    incomplete_priors = {"lengthscale": jnp.array([1.0])}
    posterior = Prior(kernel=RBF()) * Bernoulli()
    priors = prior_checks(posterior, incomplete_priors)
    assert "latent" in priors.keys()
    assert "variance" not in priors.keys()


def test_check_needless():
    complete_prior = {
        "lengthscale": tfd.Gamma(1.0, 1.0),
        "variance": tfd.Gamma(2.0, 2.0),
        "obs_noise": tfd.Gamma(3.0, 3.0),
        "latent": tfd.Normal(loc=0.0, scale=1.0),
    }
    posterior = Prior(kernel=RBF()) * Bernoulli()
    priors = prior_checks(posterior, complete_prior)
    assert priors == complete_prior
