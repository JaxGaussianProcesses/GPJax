# Copyright 2022 The GPJax Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from typing import Callable, Dict, Tuple

import jax
import distrax as dx
import jax.numpy as jnp
import jax.random as jr
import pytest
from jax.config import config
from jaxtyping import Float, Array

import gpjax as gpx
from gpjax.variational_families import (
    AbstractVariationalFamily,
    CollapsedVariationalGaussian,
    ExpectationVariationalGaussian,
    NaturalVariationalGaussian,
    VariationalGaussian,
    WhitenedVariationalGaussian,
)

# Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)


def test_abstract_variational_family():
    # Test that the abstract class cannot be instantiated.
    with pytest.raises(TypeError):
        AbstractVariationalFamily()

    # Create a dummy variational family class with abstract methods implemented.
    class DummyVariationalFamily(AbstractVariationalFamily):
        def predict(self, params: Dict, x: Float[Array, "N D"]) -> dx.Distribution:
            return dx.MultivariateNormalDiag(loc=x)

        def _initialise_params(self, key: jr.PRNGKey) -> dict:
            return {}

    # Test that the dummy variational family can be instantiated.
    dummy_variational_family = DummyVariationalFamily()
    assert isinstance(dummy_variational_family, AbstractVariationalFamily)


# Functions to test variational family parameter shapes upon initialisation.
def vector_shape(n_inducing: int) -> Tuple[int, int]:
    """Shape of a vector with n_inducing rows and 1 column"""
    return (n_inducing, 1)


def matrix_shape(n_inducing: int) -> Tuple[int, int]:
    """Shape of a matrix with n_inducing rows and 1 column"""
    return (n_inducing, n_inducing)


# Functions to test variational parameter values upon initialisation.
def vector_val(val: float) -> Callable[[int], Float[Array, "n_inducing 1"]]:
    """Vector of shape (n_inducing, 1) filled with val"""

    def vector_val_fn(n_inducing: int):
        return val * jnp.ones(vector_shape(n_inducing))

    return vector_val_fn


def diag_matrix_val(
    val: float,
) -> Callable[[int], Float[Array, "n_inducing n_inducing"]]:
    """Diagonal matrix of shape (n_inducing, n_inducing) filled with val"""

    def diag_matrix_fn(n_inducing: int) -> Float[Array, "n_inducing n_inducing"]:
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
    n_test: int,
    n_inducing: int,
    variational_family: AbstractVariationalFamily,
    moment_names: Tuple[str, str],
    shapes: Tuple,
    values: Tuple,
) -> None:

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
    assert isinstance(predictive_dist_fn, Callable)

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
def test_collapsed_variational_gaussian(
    n_test: int, n_inducing: int, n_datapoints: int, point_dim: int
) -> None:
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
    assert isinstance(params["variational_family"]["inducing_inputs"], jax.Array)

    # Test predictions
    params = variational_family._initialise_params(jr.PRNGKey(123))
    predictive_dist_fn = variational_family(params, D)
    assert isinstance(predictive_dist_fn, Callable)

    predictive_dist = predictive_dist_fn(test_inputs)
    assert isinstance(predictive_dist, dx.Distribution)

    mu = predictive_dist.mean()
    sigma = predictive_dist.covariance()

    assert isinstance(mu, jnp.ndarray)
    assert isinstance(sigma, jnp.ndarray)
    assert mu.shape == (n_test,)
    assert sigma.shape == (n_test, n_test)
