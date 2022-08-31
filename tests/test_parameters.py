import typing as tp

import distrax as dx
import jax.numpy as jnp
import jax.random as jr
import pytest
from tensorflow_probability.substrates.jax import distributions as tfd

from gpjax.gps import Prior
from gpjax.kernels import RBF
from gpjax.likelihoods import Bernoulli, Gaussian
from gpjax.parameters import (
    build_bijectors,
    copy_dict_structure,
    evaluate_priors,
    initialise,
    log_density,
    prior_checks,
    recursive_complete,
    recursive_items,
    structure_priors,
    transform,
)


#########################
# Test base functionality
#########################
@pytest.mark.parametrize("lik", [Gaussian])
def test_initialise(lik):
    key = jr.PRNGKey(123)
    posterior = Prior(kernel=RBF()) * lik(num_datapoints=10)
    params, _, _ = initialise(posterior, key).unpack()
    assert list(sorted(params.keys())) == [
        "kernel",
        "likelihood",
        "mean_function",
    ]


def test_non_conjugate_initialise():
    posterior = Prior(kernel=RBF()) * Bernoulli(num_datapoints=10)
    params, _, _ = initialise(posterior, jr.PRNGKey(123)).unpack()
    assert list(sorted(params.keys())) == [
        "kernel",
        "latent",
        "likelihood",
        "mean_function",
    ]


#########################
# Test priors
#########################
@pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
def test_lpd(x):
    val = jnp.array(x)
    dist = tfd.Normal(loc=0.0, scale=1.0)
    lpd = log_density(val, dist)
    assert lpd is not None
    assert log_density(val, None) == 0.0


@pytest.mark.parametrize("lik", [Gaussian, Bernoulli])
def test_prior_template(lik):
    posterior = Prior(kernel=RBF()) * lik(num_datapoints=10)
    params, _, _ = initialise(posterior, jr.PRNGKey(123)).unpack()
    prior_container = copy_dict_structure(params)
    for (
        k,
        v1,
        v2,
    ) in recursive_items(params, prior_container):
        assert v2 == None


@pytest.mark.parametrize("lik", [Gaussian, Bernoulli])
def test_recursive_complete(lik):
    posterior = Prior(kernel=RBF()) * lik(num_datapoints=10)
    params, _, _ = initialise(posterior, jr.PRNGKey(123)).unpack()
    priors = {"kernel": {}}
    priors["kernel"]["lengthscale"] = tfd.HalfNormal(scale=2.0)
    container = copy_dict_structure(params)
    complete_priors = recursive_complete(container, priors)
    for (
        k,
        v1,
        v2,
    ) in recursive_items(params, complete_priors):
        if k == "lengthscale":
            assert isinstance(v2, tfd.HalfNormal)
        else:
            assert v2 == None


def test_prior_evaluation():
    """
    Test the regular setup that every parameter has a corresponding prior distribution attached to its unconstrained
    value.
    """
    params = {
        "kernel": {
            "lengthscale": jnp.array([1.0]),
            "variance": jnp.array([1.0]),
        },
        "likelihood": {"obs_noise": jnp.array([1.0])},
    }
    priors = {
        "kernel": {
            "lengthscale": tfd.Gamma(1.0, 1.0),
            "variance": tfd.Gamma(2.0, 2.0),
        },
        "likelihood": {"obs_noise": tfd.Gamma(3.0, 3.0)},
    }
    lpd = evaluate_priors(params, priors)
    assert pytest.approx(lpd) == -2.0110168


def test_none_prior():
    """
    Test that multiple dispatch is working in the case of no priors.
    """
    params = {
        "kernel": {
            "lengthscale": jnp.array([1.0]),
            "variance": jnp.array([1.0]),
        },
        "likelihood": {"obs_noise": jnp.array([1.0])},
    }
    priors = copy_dict_structure(params)
    lpd = evaluate_priors(params, priors)
    assert lpd == 0.0


def test_incomplete_priors():
    """
    Test the case where a user specifies priors for some, but not all, parameters.
    """
    params = {
        "kernel": {
            "lengthscale": jnp.array([1.0]),
            "variance": jnp.array([1.0]),
        },
        "likelihood": {"obs_noise": jnp.array([1.0])},
    }
    priors = {
        "kernel": {
            "lengthscale": tfd.Gamma(1.0, 1.0),
            "variance": tfd.Gamma(2.0, 2.0),
        },
    }
    container = copy_dict_structure(params)
    complete_priors = recursive_complete(container, priors)
    lpd = evaluate_priors(params, complete_priors)
    assert pytest.approx(lpd) == -1.6137061


@pytest.mark.parametrize("num_datapoints", [1, 10])
def test_checks(num_datapoints):
    incomplete_priors = {"lengthscale": jnp.array([1.0])}
    posterior = Prior(kernel=RBF()) * Bernoulli(num_datapoints=num_datapoints)
    priors = prior_checks(incomplete_priors)
    assert "latent" in priors.keys()
    assert "variance" not in priors.keys()
    assert isinstance(priors["latent"], dx.Normal)


def test_structure_priors():
    posterior = Prior(kernel=RBF()) * Gaussian(num_datapoints=10)
    params, _, _ = initialise(posterior, jr.PRNGKey(123)).unpack()
    priors = {
        "kernel": {
            "lengthscale": tfd.Gamma(1.0, 1.0),
            "variance": tfd.Gamma(2.0, 2.0),
        },
    }
    structured_priors = structure_priors(params, priors)

    def recursive_fn(d1, d2, fn: tp.Callable[[tp.Any], tp.Any]):
        for key, value in d1.items():
            if type(value) is dict:
                yield from recursive_fn(value, d2[key], fn)
            else:
                yield fn(key, key)

    for v in recursive_fn(params, structured_priors, lambda k1, k2: k1 == k2):
        assert v


@pytest.mark.parametrize("latent_prior", [dx.Laplace(0.0, 1.0), tfd.Laplace(0.0, 1.0)])
def test_prior_checks(latent_prior):
    priors = {
        "kernel": {"lengthscale": None, "variance": None},
        "mean_function": {},
        "liklelihood": {"variance": None},
        "latent": None,
    }
    new_priors = prior_checks(priors)
    assert "latent" in new_priors.keys()
    assert new_priors["latent"].name == "Normal"

    priors = {
        "kernel": {"lengthscale": None, "variance": None},
        "mean_function": {},
        "liklelihood": {"variance": None},
    }
    new_priors = prior_checks(priors)
    assert "latent" in new_priors.keys()
    assert new_priors["latent"].name == "Normal"

    priors = {
        "kernel": {"lengthscale": None, "variance": None},
        "mean_function": {},
        "liklelihood": {"variance": None},
        "latent": latent_prior,
    }
    with pytest.warns(UserWarning):
        new_priors = prior_checks(priors)
    assert "latent" in new_priors.keys()
    assert new_priors["latent"].name == "Laplace"


#########################
# Test transforms
#########################
@pytest.mark.parametrize("num_datapoints", [1, 10])
@pytest.mark.parametrize("likelihood", [Gaussian, Bernoulli])
def test_output(num_datapoints, likelihood):
    posterior = Prior(kernel=RBF()) * likelihood(num_datapoints=num_datapoints)
    params, _, bijectors = initialise(posterior, jr.PRNGKey(123)).unpack()

    assert isinstance(bijectors, dict)
    for k, v1, v2 in recursive_items(bijectors, bijectors):
        assert isinstance(v1.forward, tp.Callable)
        assert isinstance(v2.inverse, tp.Callable)

    unconstrained_params = transform(params, bijectors, forward=False)
    assert (
        unconstrained_params["kernel"]["lengthscale"] != params["kernel"]["lengthscale"]
    )
    backconstrained_params = transform(unconstrained_params, bijectors, forward=True)
    for k, v1, v2 in recursive_items(params, unconstrained_params):
        assert v1.dtype == v2.dtype

    for k, v1, v2 in recursive_items(params, backconstrained_params):
        assert all(v1 == v2)

    augmented_params = params
    augmented_params["test_param"] = jnp.array([1.0])
    a_bijectors = build_bijectors(augmented_params)

    assert "test_param" in list(a_bijectors.keys())
    assert a_bijectors["test_param"].forward(jnp.array([1.0])) == 1.0
    assert a_bijectors["test_param"].inverse(jnp.array([1.0])) == 1.0
