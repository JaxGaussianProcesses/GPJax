import typing as tp
from mimetypes import init

import distrax as dx
import jax.numpy as jnp
import jax.random as jr
import pytest

import gpjax as gpx
from gpjax.variational_families import (
    AbstractVariationalFamily,
    CollapsedVariationalGaussian,
    ExpectationVariationalGaussian,
    NaturalVariationalGaussian,
    VariationalGaussian,
    WhitenedVariationalGaussian,
)


def test_abstract_variational_family():
    with pytest.raises(TypeError):
        AbstractVariationalFamily()


def vector_shape(n_inducing):
    """Shape of a vector with n_inducing rows and 1 column"""
    return (n_inducing, 1)


def matrix_shape(n_inducing):
    """Shape of a matrix with n_inducing rows and 1 column"""
    return (n_inducing, n_inducing)


def vector_val(val):
    """Vector of shape (n_inducing, 1) filled with val"""

    def vector_val_fn(n_inducing):
        return val * jnp.ones(vector_shape(n_inducing))

    return vector_val_fn


def diag_matrix_val(val):
    """Diagonal matrix of shape (n_inducing, n_inducing) filled with val"""

    def diag_matrix_fn(n_inducing):
        return jnp.eye(n_inducing) * val

    return diag_matrix_fn


@pytest.mark.parametrize("n_test", [1, 10])
@pytest.mark.parametrize("n_inducing", [1, 10, 20])
@pytest.mark.parametrize(
    "variational_family, moment_names, shapes, values",
    [
        (
            VariationalGaussian,
            ["variational_mean", "variational_root_covariance"],
            [vector_shape, matrix_shape],
            [vector_val(0.0), diag_matrix_val(1.0)],
        ),
        (
            WhitenedVariationalGaussian,
            ["variational_mean", "variational_root_covariance"],
            [vector_shape, matrix_shape],
            [vector_val(0.0), diag_matrix_val(1.0)],
        ),
        (
            NaturalVariationalGaussian,
            ["natural_vector", "natural_matrix"],
            [vector_shape, matrix_shape],
            [vector_val(0.0), diag_matrix_val(-0.5)],
        ),
        (
            ExpectationVariationalGaussian,
            ["expectation_vector", "expectation_matrix"],
            [vector_shape, matrix_shape],
            [vector_val(0.0), diag_matrix_val(1.0)],
        ),
    ],
)
def test_variational_gaussians(
    n_test, n_inducing, variational_family, moment_names, shapes, values
):

    # Initialise variational family:
    prior = gpx.Prior(kernel=gpx.RBF())
    inducing_inputs = jnp.linspace(-5.0, 5.0, n_inducing).reshape(-1, 1)
    test_inputs = jnp.linspace(-5.0, 5.0, n_test).reshape(-1, 1)
    q = variational_family(prior=prior, inducing_inputs=inducing_inputs)

    # Test init:
    assert q.num_inducing == n_inducing
    assert isinstance(q, AbstractVariationalFamily)

    # Test params and keys:
    params = q._initialise_params(jr.PRNGKey(123))
    assert isinstance(params, dict)

    config_params = gpx.config.get_defaults()

    # Test inducing induput parameters:
    assert "inducing_inputs" in params["variational_family"].keys()
    assert "inducing_inputs" in config_params["transformations"].keys()

    for moment_name, shape, value in zip(moment_names, shapes, values):

        moment_params = params["variational_family"]["moments"]

        assert moment_name in moment_params.keys()
        assert moment_name in config_params["transformations"].keys()

        # Test moment shape and values:
        moment = moment_params[moment_name]
        assert isinstance(moment, jnp.ndarray)
        assert moment.shape == shape(n_inducing)
        assert (moment == value(n_inducing)).all()

    # Test KL
    params = q._initialise_params(jr.PRNGKey(123))
    kl = q.prior_kl(params)
    assert isinstance(kl, jnp.ndarray)

    # Test predictions
    predictive_dist_fn = q(params)
    assert isinstance(predictive_dist_fn, tp.Callable)

    predictive_dist = predictive_dist_fn(test_inputs)
    assert isinstance(predictive_dist, dx.Distribution)

    mu = predictive_dist.mean()
    sigma = predictive_dist.covariance()

    assert isinstance(mu, jnp.ndarray)
    assert isinstance(sigma, jnp.ndarray)
    assert mu.shape == (n_test,)
    assert sigma.shape == (n_test, n_test)


@pytest.mark.parametrize("n_test", [1, 10])
@pytest.mark.parametrize("n_datapoints", [1, 10])
@pytest.mark.parametrize("n_inducing", [1, 10, 20])
@pytest.mark.parametrize("point_dim", [1, 2])
def test_collapsed_variational_gaussian(n_test, n_inducing, n_datapoints, point_dim):
    x = jnp.linspace(-5.0, 5.0, n_datapoints).reshape(-1, 1)
    y = jnp.sin(x) + jr.normal(key=jr.PRNGKey(123), shape=x.shape) * 0.1
    x = jnp.hstack([x] * point_dim)
    D = gpx.Dataset(X=x, y=y)

    prior = gpx.Prior(kernel=gpx.RBF())

    inducing_inputs = jnp.linspace(-5.0, 5.0, n_inducing).reshape(-1, 1)
    inducing_inputs = jnp.hstack([inducing_inputs] * point_dim)
    test_inputs = jnp.linspace(-5.0, 5.0, n_test).reshape(-1, 1)
    test_inputs = jnp.hstack([test_inputs] * point_dim)

    variational_family = CollapsedVariationalGaussian(
        prior=prior,
        likelihood=gpx.Gaussian(num_datapoints=D.n),
        inducing_inputs=inducing_inputs,
    )

    # We should raise an error for non-Gaussian likelihoods:
    with pytest.raises(TypeError):
        CollapsedVariationalGaussian(
            prior=prior,
            likelihood=gpx.Bernoulli(num_datapoints=D.n),
            inducing_inputs=inducing_inputs,
        )

    # Test init
    assert variational_family.num_inducing == n_inducing
    params = gpx.config.get_defaults()
    assert "inducing_inputs" in params["transformations"].keys()
    assert (variational_family.inducing_inputs == inducing_inputs).all()

    # Test params
    params = variational_family._initialise_params(jr.PRNGKey(123))
    assert isinstance(params, dict)
    assert "likelihood" in params.keys()
    assert "obs_noise" in params["likelihood"].keys()
    assert "inducing_inputs" in params["variational_family"].keys()
    assert params["variational_family"]["inducing_inputs"].shape == (
        n_inducing,
        point_dim,
    )
    assert isinstance(params["variational_family"]["inducing_inputs"], jnp.DeviceArray)

    # Test predictions
    params = variational_family._initialise_params(jr.PRNGKey(123))
    predictive_dist_fn = variational_family(D, params)
    assert isinstance(predictive_dist_fn, tp.Callable)

    predictive_dist = predictive_dist_fn(test_inputs)
    assert isinstance(predictive_dist, dx.Distribution)

    mu = predictive_dist.mean()
    sigma = predictive_dist.covariance()

    assert isinstance(mu, jnp.ndarray)
    assert isinstance(sigma, jnp.ndarray)
    assert mu.shape == (n_test,)
    assert sigma.shape == (n_test, n_test)
