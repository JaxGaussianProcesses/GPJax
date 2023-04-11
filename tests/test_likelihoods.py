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

from typing import Callable
from itertools import product

import jax.tree_util as jtu
import distrax as dx
import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.bijectors as tfb
import numpy as np
import pytest
from jax.config import config
from jax.random import KeyArray
from jaxtyping import Array, Float

from gpjax.likelihoods import (
    AbstractLikelihood,
    Bernoulli,
    Gaussian,
    inv_probit,
)

# Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)


class BaseTestLikelihood:
    """A base class that contains all tests applied on likelihoods."""

    likelihood: AbstractLikelihood
    static_fields: list[str] = ["num_datapoints"]

    def pytest_generate_tests(self, metafunc):
        """This is called automatically by pytest"""
        id_func = lambda x: "-".join([f"{k}={v}" for k, v in x.items()])
        funcarglist = metafunc.cls.params.get(metafunc.function.__name__, None)

        if funcarglist is None:
            return
        else:
            argnames = sorted(funcarglist[0])
            metafunc.parametrize(
                argnames,
                [[funcargs[name] for name in argnames] for funcargs in funcarglist],
                ids=id_func,
            )

    @pytest.mark.parametrize("n", [1, 2, 10], ids=lambda x: f"n={x}")
    def test_initialisation(self, fields: dict, n: int) -> None:

        # Input fields as JAX arrays
        fields = {k: jnp.array([v]) for k, v in fields.items()}

        # Initialise
        likelihood: AbstractLikelihood = self.likelihood(num_datapoints=n, **fields)

        # Check properties
        for field, value in fields.items():
            assert getattr(likelihood, field) == value

        # Test that pytree returns param_field objects (and not static_field)
        leaves = jtu.tree_leaves(likelihood)
        assert len(leaves) == len(set(fields) - set(self.static_fields))

        # Test dtype of params
        for v in leaves:
            assert v.dtype == jnp.float64

        # Check meta leaves
        meta = likelihood._pytree__meta
        assert not any(f in meta.keys() for f in self.static_fields)
        assert list(meta.keys()) == sorted(set(fields) - set(self.static_fields))

        for field in meta:

            # Bijectors
            if field in ["obs_noise"]:
                assert isinstance(meta[field]["bijector"], tfb.Softplus)

            # Trainability state
            assert meta[field]["trainable"] == True

    @pytest.mark.parametrize("n", [1, 2, 10], ids=lambda x: f"n={x}")
    def test_link_functions(self, n: int):

        # Initialize likelihood with defaults
        likelihood: AbstractLikelihood = self.likelihood(num_datapoints=n)

        # Create input values
        x = jnp.linspace(-3.0, 3.0).reshape(-1, 1)
        # Test likelihood link function.
        assert isinstance(likelihood.link_function, Callable)
        assert isinstance(likelihood.link_function(f), dx.Distribution)


prod = lambda inp: [
    {"fields": dict(zip(inp.keys(), values))} for values in product(*inp.values())
]


class TestGaussian(BaseTestLikelihood):
    likelihood = Gaussian
    fields = prod({"obs_noise": [0.1, 0.5, 1.0]})
    params = {"test_initialisation": fields}
    static_fields = ["num_datapoints"]


class TestBernoulli(BaseTestLikelihood):
    likelihood = Bernoulli
    fields = prod({})
    params = {"test_initialisation": fields}
    static_fields = ["num_datapoints"]


class TestAbstract(BaseTestLikelihood):
    class DummyLikelihood(AbstractLikelihood):
        def predict(self, dist: dx.Distribution) -> dx.Distribution:
            return dx.Normal(0.0, 1.0)

        def link_function(self, f: Float[Array, "N 1"]) -> Float[Array, "N 1"]:
            return dx.MultivariateNormalDiag(loc=f)

    likelihood = DummyLikelihood
    fields = prod({})
    params = {"test_initialisation": fields}
    static_fields = ["num_datapoints"]


# def test_abstract_likelihood():
#     # Test that abstract likelihoods cannot be instantiated.
#     with pytest.raises(TypeError):
#         AbstractLikelihood(num_datapoints=123)

#     # Create a dummy likelihood class with abstract methods implemented.
#     class DummyLikelihood(AbstractLikelihood):
#         def predict(self, dist: dx.Distribution) -> dx.Distribution:
#             return dx.Normal(0.0, 1.0)

#         def link_function(self, f: Float[Array, "N 1"]) -> Float[Array, "N 1"]:
#             return dx.MultivariateNormalDiag(loc=f)

#     # Test that the dummy likelihood can be instantiated.
#     dummy_likelihood = DummyLikelihood(num_datapoints=123)
#     assert isinstance(dummy_likelihood, AbstractLikelihood)


# @pytest.mark.parametrize("n", [1, 10])
# @pytest.mark.parametrize("noise", [0.1, 0.5, 1.0])
# def test_gaussian_init(n: int, noise: float) -> None:

#     likelihood = Gaussian(num_datapoints=n, obs_noise=jnp.array([noise]))

#     assert likelihood.obs_noise == jnp.array([noise])
#     assert likelihood.num_datapoints == n
#     assert jtu.tree_leaves(likelihood) == [jnp.array([noise])]


# @pytest.mark.parametrize("n", [1, 10])
# def test_beroulli_init(n: int) -> None:

#     likelihood = Bernoulli(num_datapoints=n)
#     assert likelihood.num_datapoints == n
#     assert jtu.tree_leaves(likelihood) == []


# @pytest.mark.parametrize("lik", [Gaussian, Bernoulli])
# @pytest.mark.parametrize("n", [1, 10])
# def test_link_fns(lik: AbstractLikelihood, n: int) -> None:

#     # Create function values.
#     f = jnp.linspace(-3.0, 3.0).reshape(-1, 1)

#     # Initialise likelihood.
#     likelihood = lik(num_datapoints=n)

#     # Test likelihood link function.
#     assert isinstance(likelihood.link_function, Callable)
#     assert isinstance(likelihood.link_function(f), dx.Distribution)


# @pytest.mark.parametrize("noise", [0.1, 0.5, 1.0])
# @pytest.mark.parametrize("n", [1, 2, 10])
# def test_call_gaussian(noise: float, n: int) -> None:
#     key = jr.PRNGKey(123)

#     # Initialise likelihood and parameters.
#     likelihood = Gaussian(num_datapoints=n, obs_noise=jnp.array([noise]))

#     # Construct latent function distribution.
#     latent_mean = jr.uniform(key, shape=(n,))
#     latent_sqrt = jr.uniform(key, shape=(n, n))
#     latent_cov = jnp.matmul(latent_sqrt, latent_sqrt.T)
#     latent_dist = dx.MultivariateNormalFullCovariance(latent_mean, latent_cov)

#     # Test call method.
#     pred_dist = likelihood(latent_dist)

#     # Check that the distribution is a MultivariateNormalFullCovariance.
#     assert isinstance(pred_dist, dx.MultivariateNormalFullCovariance)

#     # Check predictive mean and variance.
#     assert (pred_dist.mean() == latent_mean).all()

#     noise_matrix = jnp.eye(n) * noise
#     assert np.allclose(
#         pred_dist.scale_tri, jnp.linalg.cholesky(latent_cov + noise_matrix)
#     )


# @pytest.mark.parametrize("n", [1, 2, 10])
# def test_call_bernoulli(n: int) -> None:
#     key = jr.PRNGKey(123)

#     # Initialise likelihood and parameters.
#     likelihood = Bernoulli(num_datapoints=n)

#     # Construct latent function distribution.
#     latent_mean = jr.uniform(key, shape=(n,))
#     latent_sqrt = jr.uniform(key, shape=(n, n))
#     latent_cov = jnp.matmul(latent_sqrt, latent_sqrt.T)
#     latent_dist = dx.MultivariateNormalFullCovariance(latent_mean, latent_cov)

#     # Test call method.
#     pred_dist = likelihood(latent_dist)

#     # Check that the distribution is a Bernoulli.
#     assert isinstance(pred_dist, dx.Bernoulli)

#     # Check predictive mean and variance.

#     p = inv_probit(latent_mean / jnp.sqrt(1.0 + jnp.diagonal(latent_cov)))
#     assert (pred_dist.mean() == p).all()
#     assert (pred_dist.variance() == p * (1.0 - p)).all()
