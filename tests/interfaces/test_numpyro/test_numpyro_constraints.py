import chex
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro.distributions import constraints

from gpjax.gps import Prior
from gpjax.interfaces.numpyro import add_constraints, numpyro_dict_params
from gpjax.kernels import RBF
from gpjax.likelihoods import Gaussian
from gpjax.parameters import initialise


# TODO: test conjugate posterior
def _get_conjugate_posterior_params() -> dict:
    kernel = RBF()
    prior = Prior(kernel=kernel)
    lik = Gaussian()
    posterior = prior * lik
    params = initialise(posterior)
    return params


@pytest.mark.parametrize(
    "input_types",
    [
        str(1.0),
        np.array(1.0),
        None,
    ],
)
def test_numpyro_dict_params_defaults_nullcase(input_types):

    demo_params = {
        "lengthscale": input_types,
        "variance": input_types,
        "obs_noise": input_types,
    }
    with pytest.raises(ValueError):

        numpyro_dict_params(demo_params)


def test_numpyro_dict_params_defaults_array():

    demo_params = {
        "lengthscale": jnp.array(1.0),
        "variance": jnp.array(1.0),
        "obs_noise": jnp.array(1.0),
    }

    numpyro_params = numpyro_dict_params(demo_params)

    assert set(numpyro_params) == set(demo_params.keys())
    for ikey, iparam in demo_params.items():
        # check keys exist for param
        assert set(numpyro_params[ikey].keys()) == set(
            ("init_value", "constraint", "param_type")
        )
        # check init value is the same as initial value
        chex.assert_equal(numpyro_params[ikey]["init_value"], iparam)
        # check default constraint is positive
        chex.assert_equal(numpyro_params[ikey]["constraint"], constraints.positive)
        # check if param type is param
        chex.assert_equal(numpyro_params[ikey]["param_type"], "param")

    # check we didn't modify original dictionary
    chex.assert_equal(
        demo_params,
        {
            "lengthscale": jnp.array(1.0),
            "variance": jnp.array(1.0),
            "obs_noise": jnp.array(1.0),
        },
    )


def test_numpyro_dict_params_defaults_float():

    demo_params = {
        "lengthscale": 1.0,
        "variance": 1.0,
        "obs_noise": 1.0,
    }

    numpyro_params = numpyro_dict_params(demo_params)

    assert set(numpyro_params) == set(demo_params.keys())
    for ikey, iparam in demo_params.items():
        # check keys exist for param
        assert set(numpyro_params[ikey].keys()) == set(
            ("init_value", "constraint", "param_type")
        )
        # check init value is the same as initial value
        chex.assert_equal(numpyro_params[ikey]["init_value"], iparam)
        # check default constraint is positive
        chex.assert_equal(numpyro_params[ikey]["constraint"], constraints.positive)
        # check if param type is param
        chex.assert_equal(numpyro_params[ikey]["param_type"], "param")

    # check we didn't modify original dictionary
    chex.assert_equal(
        demo_params,
        {
            "lengthscale": 1.0,
            "variance": 1.0,
            "obs_noise": 1.0,
        },
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
def test_numpyro_add_constraints_all(constraint):

    gpjax_params = _get_conjugate_posterior_params()
    numpyro_params = numpyro_dict_params(gpjax_params)

    # add constraint
    new_numpyro_params = add_constraints(numpyro_params, constraint)
    for iparams in new_numpyro_params.values():

        # check if constraint in new dictionary
        chex.assert_equal(iparams["constraint"], constraint)

    # check we didn't modify original dictionary
    chex.assert_equal(gpjax_params, _get_conjugate_posterior_params())


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

    gpjax_params = _get_conjugate_posterior_params()
    numpyro_params = numpyro_dict_params(gpjax_params)

    # add constraint
    new_numpyro_params = add_constraints(numpyro_params, variable, constraint)

    # check if constraint in new dictionary
    chex.assert_equal(new_numpyro_params[variable]["constraint"], constraint)

    # check we didn't modify original dictionary
    chex.assert_equal(gpjax_params, _get_conjugate_posterior_params())


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
def test_numpyro_add_constraints_dict(variable, constraint):

    gpjax_params = _get_conjugate_posterior_params()
    numpyro_params = numpyro_dict_params(gpjax_params)

    # create new dictionary
    new_param_dict = {str(variable): constraint}

    # add constraint
    new_numpyro_params = add_constraints(numpyro_params, new_param_dict)

    # check if constraint in new dictionary
    chex.assert_equal(new_numpyro_params[variable]["constraint"], constraint)

    # check we didn't modify original dictionary
    chex.assert_equal(gpjax_params, _get_conjugate_posterior_params())
