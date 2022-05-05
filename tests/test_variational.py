import pytest
import gpjax as gpx
import jax.numpy as jnp
import jax.random as jr
import jax


def test_abstract_variational_family():
    with pytest.raises(TypeError):
        gpx.variational.VariationalFamily()


@pytest.mark.parametrize('n_inducing', [1, 10, 20])
def test_variational_gaussian_post_init(n_inducing):
    inducing_points = jnp.linspace(-3., 3., n_inducing).reshape(-1, 1)
    variational_family = gpx.variational.VariationalGaussian(inducing_inputs = inducing_points)

    assert variational_family.num_inducing == n_inducing

    assert jnp.sum(variational_family.variational_mean) == 0.
    assert variational_family.variational_mean.shape == (n_inducing, 1)

    assert variational_family.variational_root_covariance.shape == (n_inducing, n_inducing)


@pytest.mark.parametrize('n_inducing', [1, 10, 20])
def test_variational_gaussian_params(n_inducing):
    inducing_points = jnp.linspace(-3., 3., n_inducing).reshape(-1, 1)
    variational_family = gpx.variational.VariationalGaussian(inducing_inputs = inducing_points)

    params = variational_family.params
    assert isinstance(params, dict)
    assert 'inducing_inputs' in params.keys()
    assert 'variational_mean' in params.keys()
    assert 'variational_root_covariance' in params.keys()

    assert params['inducing_inputs'].shape == (n_inducing, 1)
    assert params['variational_mean'].shape == (n_inducing, 1)
    assert params['variational_root_covariance'].shape == (n_inducing, n_inducing)

    assert isinstance(params['inducing_inputs'], jnp.DeviceArray)
    assert isinstance(params['variational_mean'], jnp.DeviceArray)
    assert isinstance(params['variational_root_covariance'], jnp.DeviceArray)