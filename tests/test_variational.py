from statistics import mode

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import py
import pyexpat
import pytest
from chex import assert_max_traces

import gpjax as gpx


def test_abstract_variational_family():
    with pytest.raises(TypeError):
        gpx.variational.VariationalFamily()


@pytest.mark.parametrize("diag", [True, False])
@pytest.mark.parametrize("n_inducing", [1, 10, 20])
def test_variational_gaussian_diag(diag, n_inducing):
    prior = gpx.Prior(kernel=gpx.RBF())
    inducing_points = jnp.linspace(-3.0, 3.0, n_inducing).reshape(-1, 1)
    variational_family = gpx.variational.VariationalGaussian(
        prior=prior,
        inducing_inputs=inducing_points, 
        diag=diag
    )

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


@pytest.mark.parametrize("n_inducing", [1, 10, 20])
def test_variational_gaussian_params(n_inducing):
    prior = gpx.Prior(kernel=gpx.RBF())
    inducing_points = jnp.linspace(-3.0, 3.0, n_inducing).reshape(-1, 1)
    variational_family = gpx.variational.VariationalGaussian(
        prior=prior,
        inducing_inputs=inducing_points
    )

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


# @pytest.mark.parametrize("n, n_inducing", [(10, 2), (50, 10)])
# def test_variational_posterior(n, n_inducing):
#     x = jnp.linspace(-5., 5., n).reshape(-1, 1)
#     y = jnp.sin(x)
#     D = gpx.Dataset(X=x, y=y)
#     inducing_points = jnp.linspace(-3.0, 3.0, n_inducing).reshape(-1, 1)
#     prior = gpx.Prior(kernel = gpx.RBF())
#     lik =gpx.likelihoods.Gaussian(num_datapoints = n)
#     p = prior  * lik
#     q = gpx.variational.VariationalGaussian(inducing_inputs = inducing_points)

#     model = gpx.variational.VariationalPosterior(p, q)

#     assert model.posterior.prior == prior
#     assert model.posterior.likelihood == lik
