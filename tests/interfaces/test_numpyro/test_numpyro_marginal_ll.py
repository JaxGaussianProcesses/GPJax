import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro.contrib.tfp import distributions as tfd
from numpyro.distributions import constraints

from gpjax.gps import Prior
from gpjax.interfaces.numpyro import (
    add_constraints,
    add_priors,
    numpyro_dict_params,
    numpyro_marginal_ll,
)
from gpjax.kernels import RBF
from gpjax.likelihoods import Gaussian
from gpjax.parameters import initialise
from gpjax.types import Dataset

seed = 123
KEY = jr.PRNGKey(seed)

# TODO: test conjugate posterior
def _get_conjugate_posterior_params() -> dict:
    kernel = RBF()
    prior = Prior(kernel=kernel)
    lik = Gaussian()
    posterior = prior * lik
    params = initialise(posterior)
    return params, posterior


def _gen_training_data(n_samples, n_features, n_latents):

    x = jr.normal(key=KEY, shape=(n_samples, n_features))
    y = jr.normal(key=KEY, shape=(n_samples, n_latents))

    return Dataset(X=x, y=y)


def test_numpyro_marginal_ll_params():

    # create sample data
    ds = _gen_training_data(10, 10, 2)

    # initialize parameters
    params, posterior = _get_conjugate_posterior_params()

    # convert to numpyro-style params
    numpyro_params = numpyro_dict_params(params)

    # initialize numpyro-style GP model
    npy_model = numpyro_marginal_ll(posterior, numpyro_params)

    # do one forward pass with context
    with numpyro.handlers.seed(rng_seed=KEY):
        model_params = numpyro.handlers.trace(npy_model).get_trace(ds)

    assert set(numpyro_params) <= set(model_params)


@pytest.mark.parametrize(
    "n_samples",
    [1, 10, 100],
)
@pytest.mark.parametrize(
    "n_features",
    [1, 10, 100],
)
@pytest.mark.parametrize(
    "n_latents",
    [
        1,
    ],
)
def test_numpyro_marginal_ll_params_shape(n_samples, n_features, n_latents):

    # create sample data
    ds = _gen_training_data(n_samples, n_features, n_latents)

    # initialize parameters
    params, posterior = _get_conjugate_posterior_params()

    # convert to numpyro-style params
    numpyro_params = numpyro_dict_params(params)

    # initialize numpyro-style GP model
    npy_model = numpyro_marginal_ll(posterior, numpyro_params)

    # do one forward pass with context
    with numpyro.handlers.seed(rng_seed=KEY):
        pred = npy_model(ds)

        chex.assert_equal_shape([ds.y.squeeze(), pred])


@pytest.mark.parametrize(
    "n_samples",
    [1, 10, 100],
)
@pytest.mark.parametrize(
    "n_features",
    [1, 10, 100],
)
@pytest.mark.parametrize(
    "n_latents",
    [
        1,
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        jnp.float32,
        jnp.float64,
    ],
)
def test_numpyro_marginal_ll_params_type(n_samples, n_features, n_latents, dtype):

    # create sample data
    ds = _gen_training_data(n_samples, n_features, n_latents)

    # convert to dtype
    ds = jax.tree_util.tree_map(lambda x: x.astype(dtype), ds)

    # initialize parameters
    params, posterior = _get_conjugate_posterior_params()

    # convert to numpyro-style params
    numpyro_params = numpyro_dict_params(params)

    # initialize numpyro-style GP model
    npy_model = numpyro_marginal_ll(posterior, numpyro_params)

    # do one forward pass with context
    with numpyro.handlers.seed(rng_seed=KEY):
        pred = npy_model(ds)

        chex.assert_equal(pred.dtype, ds.y.dtype)


def test_numpyro_marginal_ll_numpyro_priors():

    # create sample data
    ds = _gen_training_data(10, 10, 2)

    # initialize parameters
    params, posterior = _get_conjugate_posterior_params()

    # convert to numpyro-style params
    numpyro_params = numpyro_dict_params(params)

    # convert to priors
    numpyro_params = add_priors(numpyro_params, dist.LogNormal(0.0, 10.0))

    # initialize numpyro-style GP model
    npy_model = numpyro_marginal_ll(posterior, numpyro_params)

    # do one forward pass with context
    with numpyro.handlers.seed(rng_seed=KEY):
        model_params = numpyro.handlers.trace(npy_model).get_trace(ds)

    assert set(numpyro_params) <= set(model_params)


@pytest.mark.parametrize(
    "n_samples",
    [1, 10, 100],
)
@pytest.mark.parametrize(
    "n_features",
    [1, 10, 100],
)
@pytest.mark.parametrize(
    "n_latents",
    [
        1,
    ],
)
def test_numpyro_marginal_ll_numpyro_priors_shape(n_samples, n_features, n_latents):

    # create sample data
    ds = _gen_training_data(n_samples, n_features, n_latents)

    # initialize parameters
    params, posterior = _get_conjugate_posterior_params()

    # convert to numpyro-style params
    numpyro_params = numpyro_dict_params(params)

    # convert to priors
    numpyro_params = add_priors(numpyro_params, dist.LogNormal(0.0, 10.0))

    # initialize numpyro-style GP model
    npy_model = numpyro_marginal_ll(posterior, numpyro_params)

    # do one forward pass with context
    with numpyro.handlers.seed(rng_seed=KEY):
        pred = npy_model(ds)

        chex.assert_equal_shape([pred, ds.y.squeeze()])


@pytest.mark.parametrize(
    "n_samples",
    [1, 10, 100],
)
@pytest.mark.parametrize(
    "n_features",
    [1, 10, 100],
)
@pytest.mark.parametrize(
    "n_latents",
    [
        1,
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        jnp.float32,
        jnp.float64,
    ],
)
def test_numpyro_marginal_ll_numpyro_priors_type(n_samples, n_features, n_latents, dtype):

    # create sample data
    ds = _gen_training_data(n_samples, n_features, n_latents)

    # convert to tyle
    ds = jax.tree_util.tree_map(lambda x: x.astype(dtype), ds)

    # initialize parameters
    params, posterior = _get_conjugate_posterior_params()

    # convert to numpyro-style params
    numpyro_params = numpyro_dict_params(params)

    # convert to priors
    numpyro_params = add_priors(numpyro_params, dist.LogNormal(0.0, 10.0))

    # initialize numpyro-style GP model
    npy_model = numpyro_marginal_ll(posterior, numpyro_params)

    # do one forward pass with context
    with numpyro.handlers.seed(rng_seed=KEY):
        pred = npy_model(ds)

        chex.assert_equal(pred.dtype, ds.y.dtype)


@pytest.mark.parametrize(
    "n_samples",
    [1, 10, 100],
)
@pytest.mark.parametrize(
    "n_features",
    [1, 10, 100],
)
@pytest.mark.parametrize(
    "n_latents",
    [
        1,
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        jnp.float32,
        jnp.float64,
    ],
)
def test_numpyro_marginal_ll_tfp_priors_type(n_samples, n_features, n_latents, dtype):

    # create sample data
    ds = _gen_training_data(n_samples, n_features, n_latents)

    # convert to tyle
    ds = jax.tree_util.tree_map(lambda x: x.astype(dtype), ds)

    # initialize parameters
    params, posterior = _get_conjugate_posterior_params()

    # convert to numpyro-style params
    numpyro_params = numpyro_dict_params(params)

    # convert to priors
    numpyro_params = add_priors(numpyro_params, tfd.LogNormal(0.0, 10.0))

    # initialize numpyro-style GP model
    npy_model = numpyro_marginal_ll(posterior, numpyro_params)

    # do one forward pass with context
    with numpyro.handlers.seed(rng_seed=KEY):
        pred = npy_model(ds)

        chex.assert_equal(pred.dtype, ds.y.dtype)
