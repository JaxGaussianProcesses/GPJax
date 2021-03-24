from gpjax.interfaces.numpyro import add_constraints, numpyro_dict_params
import pytest

import jax.numpy as jnp
import jax.random as jr

import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
import chex

from gpjax.gps import Prior
from gpjax.kernels import RBF
from gpjax.likelihoods import Gaussian
from gpjax.parameters import initialise


# TODO: test conjugate posterior
def get_conjugate_posterior_params() -> dict:
    kernel = RBF()
    prior = Prior(kernel=kernel)
    lik = Gaussian()
    posterior = prior * lik
    params = initialise(posterior)
    return params


def get_numpyro_params() -> dict:

    params = {
        "lengthscale": {
            "param_type": "param",
            "init_value": jnp.array(1.0),
            "constraint": constraints.positive,
        },
        "variance": {
            "param_type": "param",
            "init_value": jnp.array(1.0),
            "constraint": constraints.positive,
        },
        "obs_noise": {
            "param_type": "param",
            "init_value": jnp.array(1.0),
            "constraint": constraints.positive,
        },
    }

    return params


def get_numpyro_priors() -> dict:

    hyperpriors = {
        "lengthscale": dist.Gamma(1.0, 1.0),
        "variance": dist.HalfCauchy(scale=1.0),
        "obs_noise": dist.HalfCauchy(scale=5.0),
    }

    return hyperpriors


def test_numpyro_dict_params_defaults():

    gpjax_params = get_conjugate_posterior_params()
    numpyro_params = numpyro_dict_params(gpjax_params)

    assert set(numpyro_params) == set(gpjax_params.keys())
    for ikey, iparam in gpjax_params.items():
        # check keys exist for param
        assert set(numpyro_params[ikey].keys()) == set(
            ("init_value", "constraint", "param_type")
        )
        # check init value is the same as initial value
        chex.assert_equal(numpyro_params[ikey]["init_value"], iparam)
        chex.assert_equal(numpyro_params[ikey]["constraint"], constraints.positive)
        chex.assert_equal(numpyro_params[ikey]["param_type"], "param")

    # check we didn't modify original dictionary
    chex.assert_equal(gpjax_params, get_conjugate_posterior_params())


@pytest.mark.parametrize(
    "variable",
    ["lengthscale", "obs_noise", "variance"],
)
@pytest.mark.parametrize(
    "constraint",
    [
        constraints.positive,
        constraints.real,
        constraints.softplus_positive,
        constraints.softplus_lower_cholesky,
    ],
)
def test_numpyro_add_constraints_str(variable, constraint):

    gpjax_params = get_conjugate_posterior_params()
    numpyro_params = numpyro_dict_params(gpjax_params)

    # add constraint
    new_numpyro_params = add_constraints(numpyro_params, variable, constraint)
    chex.assert_equal(new_numpyro_params[variable]["constraint"], constraint)
