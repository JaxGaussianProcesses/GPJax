import typing as tp

import distrax as dx

import jax.numpy as jnp
import jax.random as jr
import pytest

import gpjax as gpx


def test_abstract_variational_family():
    with pytest.raises(TypeError):
        gpx.variational_families.AbstractVariationalFamily()


@pytest.mark.parametrize("diag", [True, False])
@pytest.mark.parametrize("n_test", [1, 10])
@pytest.mark.parametrize("whiten", [True, False])
@pytest.mark.parametrize("n_inducing", [1, 10, 20])
def test_variational_gaussian(diag, n_inducing, n_test, whiten):
    prior = gpx.Prior(kernel=gpx.RBF())

    inducing_inputs = jnp.linspace(-5.0, 5.0, n_inducing).reshape(-1, 1)
    test_inputs = jnp.linspace(-5.0, 5.0, n_test).reshape(-1, 1)

    if whiten is True:
        variational_family = gpx.WhitenedVariationalGaussian(
            prior=prior, inducing_inputs=inducing_inputs, diag=diag
        )
    else:
        variational_family = gpx.VariationalGaussian(
            prior=prior, inducing_inputs=inducing_inputs, diag=diag
        )

    # Test init
    assert variational_family.num_inducing == n_inducing

    assert jnp.sum(variational_family.variational_mean) == 0.0
    assert variational_family.variational_mean.shape == (n_inducing, 1)

    assert variational_family.variational_root_covariance.shape == (
        n_inducing,
        n_inducing,
    )
    assert jnp.all(jnp.diag(variational_family.variational_root_covariance) == 1.0)

    params = gpx.config.get_defaults()
    assert "variational_root_covariance" in params["transformations"].keys()
    assert "variational_mean" in params["transformations"].keys()

    assert (variational_family.variational_root_covariance == jnp.eye(n_inducing)).all()
    assert (variational_family.variational_mean == jnp.zeros((n_inducing, 1))).all()

    # Test params
    params = variational_family._initialise_params(jr.PRNGKey(123))
    assert isinstance(params, dict)
    assert "inducing_inputs" in params["variational_family"].keys()
    assert "variational_mean" in params["variational_family"].keys()
    assert "variational_root_covariance" in params["variational_family"].keys()

    assert params["variational_family"]["inducing_inputs"].shape == (n_inducing, 1)
    assert params["variational_family"]["variational_mean"].shape == (n_inducing, 1)
    assert params["variational_family"]["variational_root_covariance"].shape == (
        n_inducing,
        n_inducing,
    )

    assert isinstance(params["variational_family"]["inducing_inputs"], jnp.DeviceArray)
    assert isinstance(params["variational_family"]["variational_mean"], jnp.DeviceArray)
    assert isinstance(
        params["variational_family"]["variational_root_covariance"], jnp.DeviceArray
    )

    params = gpx.config.get_defaults()
    assert "variational_root_covariance" in params["transformations"].keys()
    assert "variational_mean" in params["transformations"].keys()

    assert (variational_family.variational_root_covariance == jnp.eye(n_inducing)).all()
    assert (variational_family.variational_mean == jnp.zeros((n_inducing, 1))).all()

    # Test KL
    params = variational_family._initialise_params(jr.PRNGKey(123))
    kl = variational_family.prior_kl(params)
    assert isinstance(kl, jnp.ndarray)

    # Test predictions
    predictive_dist_fn = variational_family(params)
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

    variational_family = gpx.variational_families.CollapsedVariationalGaussian(
        prior=prior,
        likelihood=gpx.Gaussian(num_datapoints=D.n),
        inducing_inputs=inducing_inputs,
    )

    # We should raise an error for non-Gaussian likelihoods:
    with pytest.raises(TypeError):
        gpx.variational_families.CollapsedVariationalGaussian(
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
@pytest.mark.parametrize("n_inducing", [1, 10, 20])
def test_natural_variational_gaussian_params(n_inducing):
    prior = gpx.Prior(kernel=gpx.RBF())
    inducing_points = jnp.linspace(-3.0, 3.0, n_inducing).reshape(-1, 1)
    variational_family = gpx.variational_families.NaturalVariationalGaussian(
        prior=prior,
        inducing_inputs=inducing_points
    )

    params = variational_family.params
    assert isinstance(params, dict)
    assert "inducing_inputs" in params["variational_family"].keys()
    assert "natural_vector" in params["variational_family"].keys()
    assert "natural_matrix" in params["variational_family"].keys()

    assert params["variational_family"]["inducing_inputs"].shape == (n_inducing, 1)
    assert params["variational_family"]["natural_vector"].shape == (n_inducing, 1)
    assert params["variational_family"]["natural_matrix"].shape == (n_inducing, n_inducing)

    assert isinstance(params["variational_family"]["inducing_inputs"], jnp.DeviceArray)
    assert isinstance(params["variational_family"]["natural_vector"], jnp.DeviceArray)
    assert isinstance(params["variational_family"]["natural_matrix"], jnp.DeviceArray)
   
    params = gpx.config.get_defaults()
    assert "natural_vector" in params["transformations"].keys()
    assert "natural_matrix" in params["transformations"].keys()

    assert (variational_family.natural_matrix == -.5 * jnp.eye(n_inducing)).all()
    assert (variational_family.natural_vector == jnp.zeros((n_inducing, 1))).all()
