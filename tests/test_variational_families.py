import typing as tp

import distrax as dx

import jax.numpy as jnp
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
        variational_family = gpx.WhitenedVariationalGaussian(prior = prior,
        inducing_inputs=inducing_inputs, diag=diag
        )
    else:
        variational_family = gpx.VariationalGaussian(prior = prior,
        inducing_inputs=inducing_inputs, diag=diag
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

    #Test params
    params = variational_family.params
    assert isinstance(params, dict)
    assert "inducing_inputs" in params["variational_family"].keys()
    assert "variational_mean" in params["variational_family"].keys()
    assert "variational_root_covariance" in params["variational_family"].keys()

    assert params["variational_family"]["inducing_inputs"].shape == (n_inducing, 1)
    assert params["variational_family"]["variational_mean"].shape == (n_inducing, 1)
    assert params["variational_family"]["variational_root_covariance"].shape == (n_inducing, n_inducing)

    assert isinstance(params["variational_family"]["inducing_inputs"], jnp.DeviceArray)
    assert isinstance(params["variational_family"]["variational_mean"], jnp.DeviceArray)
    assert isinstance(params["variational_family"]["variational_root_covariance"], jnp.DeviceArray)

    params = gpx.config.get_defaults()
    assert "variational_root_covariance" in params["transformations"].keys()
    assert "variational_mean" in params["transformations"].keys()

    assert (variational_family.variational_root_covariance == jnp.eye(n_inducing)).all()
    assert (variational_family.variational_mean == jnp.zeros((n_inducing, 1))).all()
    
    #Test KL
    params = variational_family.params
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
@pytest.mark.parametrize("n_inducing", [1, 10, 20])
def test_natural_variational_gaussian(n_inducing, n_test):
    prior = gpx.Prior(kernel=gpx.RBF())
    
    inducing_inputs = jnp.linspace(-5.0, 5.0, n_inducing).reshape(-1, 1)
    test_inputs = jnp.linspace(-5.0, 5.0, n_test).reshape(-1, 1)


    variational_family = gpx.variational_families.NaturalVariationalGaussian(
        prior=prior,
        inducing_inputs=inducing_inputs
    )

    # Test init
    assert variational_family.num_inducing == n_inducing

    assert jnp.sum(variational_family.natural_vector) == 0.0
    assert variational_family.natural_vector.shape == (n_inducing, 1)

    assert variational_family.natural_matrix.shape == (
        n_inducing,
        n_inducing,
    )
    assert jnp.all(jnp.diag(variational_family.natural_matrix) == -.5)

    params = gpx.config.get_defaults()
    assert "variational_root_covariance" in params["transformations"].keys()
    assert "variational_mean" in params["transformations"].keys()

    assert (variational_family.natural_matrix == -.5 * jnp.eye(n_inducing)).all()
    assert (variational_family.natural_vector == jnp.zeros((n_inducing, 1))).all()

    # params
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


    #Test KL
    params = variational_family.params
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
