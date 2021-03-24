import chex
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro.contrib.tfp import distributions as tfd
from numpyro.distributions import constraints

from gpjax.gps import Prior
from gpjax.interfaces.numpyro import add_constraints, add_priors, numpyro_dict_params
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


def test_numpyro_dict_priors_defaults_numpyro():

    demo_priors = {
        "lengthscale": dist.LogNormal(loc=0.0, scale=1.0),
        "variance": dist.LogNormal(loc=0.0, scale=1.0),
        "obs_noise": dist.LogNormal(loc=0.0, scale=1.0),
    }

    numpyro_params = numpyro_dict_params(demo_priors)

    assert set(numpyro_params) == set(demo_priors.keys())
    for ikey, iparam in demo_priors.items():
        # check keys exist for param
        assert set(numpyro_params[ikey].keys()) == set(("prior", "param_type"))
        # check init value is the same as initial value
        chex.assert_equal(numpyro_params[ikey]["prior"], iparam)


def test_numpyro_dict_priors_defaults_tfp():

    demo_priors = {
        "lengthscale": tfd.LogNormal(loc=0.0, scale=1.0),
        "variance": tfd.LogNormal(loc=0.0, scale=1.0),
        "obs_noise": tfd.LogNormal(loc=0.0, scale=1.0),
    }

    numpyro_params = numpyro_dict_params(demo_priors)

    assert set(numpyro_params) == set(demo_priors.keys())
    for ikey, iparam in demo_priors.items():
        # check keys exist for param
        assert set(numpyro_params[ikey].keys()) == set(("prior", "param_type"))
        # check init value is the same as initial value
        chex.assert_equal(numpyro_params[ikey]["prior"], iparam)


@pytest.mark.parametrize(
    "prior",
    [
        dist.Gamma(concentration=1.0, rate=1.0),
        dist.HalfCauchy(scale=1.0),
        dist.LogNormal(loc=0.0, scale=1.0),
        tfd.Gamma(concentration=1.0, rate=1.0),
        tfd.HalfCauchy(loc=0.0, scale=1.0),
        tfd.LogNormal(loc=0.0, scale=1.0),
    ],
)
def test_numpyro_add_priors_all(prior):

    gpjax_params = _get_conjugate_posterior_params()
    numpyro_params = numpyro_dict_params(gpjax_params)

    # add constraint
    new_numpyro_params = add_priors(numpyro_params, prior)
    for iparams in new_numpyro_params.values():

        # check if constraint in new dictionary
        chex.assert_equal(iparams["param_type"], "prior")
        chex.assert_equal(iparams["prior"], prior)

    # check we didn't modify original dictionary
    chex.assert_equal(gpjax_params, _get_conjugate_posterior_params())


@pytest.mark.parametrize(
    "variable",
    ["lengthscale", "obs_noise", "variance"],
)
@pytest.mark.parametrize(
    "prior",
    [
        dist.Gamma(concentration=1.0, rate=1.0),
        dist.HalfCauchy(scale=1.0),
        dist.LogNormal(loc=0.0, scale=1.0),
        tfd.Gamma(concentration=1.0, rate=1.0),
        tfd.HalfCauchy(loc=0.0, scale=1.0),
        tfd.LogNormal(loc=0.0, scale=1.0),
    ],
)
def test_numpyro_add_priors_str(variable, prior):

    gpjax_params = _get_conjugate_posterior_params()
    numpyro_params = numpyro_dict_params(gpjax_params)

    # add constraint
    new_numpyro_params = add_priors(numpyro_params, variable, prior)

    # check if constraint in new dictionary
    chex.assert_equal(new_numpyro_params[variable]["param_type"], "prior")
    chex.assert_equal(new_numpyro_params[variable]["prior"], prior)

    # check we didn't modify original dictionary
    chex.assert_equal(gpjax_params, _get_conjugate_posterior_params())


@pytest.mark.parametrize(
    "variable",
    ["lengthscale", "obs_noise", "variance"],
)
@pytest.mark.parametrize(
    "prior",
    [
        dist.Gamma(concentration=1.0, rate=1.0),
        dist.HalfCauchy(scale=1.0),
        dist.LogNormal(loc=0.0, scale=1.0),
        tfd.Gamma(concentration=1.0, rate=1.0),
        tfd.HalfCauchy(loc=0.0, scale=1.0),
        tfd.LogNormal(loc=0.0, scale=1.0),
    ],
)
def test_numpyro_add_priors_dict(variable, prior):

    gpjax_params = _get_conjugate_posterior_params()
    numpyro_params = numpyro_dict_params(gpjax_params)

    # create new dictionary
    new_param_dict = {str(variable): prior}

    # add constraint
    new_numpyro_params = add_priors(numpyro_params, new_param_dict)

    # check if constraint in new dictionary
    chex.assert_equal(new_numpyro_params[variable]["param_type"], "prior")
    chex.assert_equal(new_numpyro_params[variable]["prior"], prior)

    # check we didn't modify original dictionary
    chex.assert_equal(gpjax_params, _get_conjugate_posterior_params())
