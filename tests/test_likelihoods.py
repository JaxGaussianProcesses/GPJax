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

from dataclasses import is_dataclass
from itertools import product
from typing import (
    Callable,
    List,
)

from jax import config
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jaxtyping import (
    Array,
    Float,
)
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd

from gpjax.likelihoods import (
    AbstractLikelihood,
    Bernoulli,
    Gaussian,
    Poisson,
    inv_probit,
)

# Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)
_initialise_key = jr.key(123)


class BaseTestLikelihood:
    """A base class that contains all tests applied on likelihoods."""

    likelihood: AbstractLikelihood
    static_fields: List[str] = ["num_datapoints"]

    def pytest_generate_tests(self, metafunc):
        """This is called automatically by pytest."""

        # function for pretty test name
        def id_func(x):
            return "-".join([f"{k}={v}" for k, v in x.items()])

        # get arguments for the test function
        funcarglist = metafunc.cls.params.get(metafunc.function.__name__, None)

        if funcarglist is None:
            return
        else:
            # equivalent of pytest.mark.parametrize applied on the metafunction
            metafunc.parametrize("fields", funcarglist, ids=id_func)

    @pytest.mark.parametrize("n", [1, 2, 10], ids=lambda x: f"n={x}")
    def test_initialisation(self, fields: dict, n: int) -> None:
        # Check that likelihood is a dataclass
        assert is_dataclass(self.likelihood)

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
        assert not any(f in meta for f in self.static_fields)
        assert list(meta.keys()) == sorted(set(fields) - set(self.static_fields))

        for field in meta:
            # Bijectors
            if field in ["obs_stddev"]:
                assert isinstance(meta[field]["bijector"], tfb.Softplus)

            # Trainability state
            assert meta[field]["trainable"] is True

    @pytest.mark.parametrize("n", [1, 2, 10], ids=lambda x: f"n={x}")
    def test_link_functions(self, n: int):
        # Initialize likelihood with defaults
        likelihood: AbstractLikelihood = self.likelihood(num_datapoints=n)

        # Create input values
        x = jnp.linspace(-3.0, 3.0).reshape(-1, 1)

        # Test likelihood link function.
        assert isinstance(likelihood.link_function, Callable)
        assert isinstance(likelihood.link_function(x), tfd.Distribution)

    @pytest.mark.parametrize("n", [1, 2, 10], ids=lambda x: f"n={x}")
    def test_call(self, fields: dict, n: int):
        # Input fields as JAX arrays
        fields = {k: jnp.array([v]) for k, v in fields.items()}

        # Initialise
        likelihood: AbstractLikelihood = self.likelihood(num_datapoints=n, **fields)

        # Construct latent function distribution.
        k1, k2 = jr.split(_initialise_key)
        latent_mean = jr.uniform(k1, shape=(n,))
        latent_sqrt = jr.uniform(k2, shape=(n, n))
        latent_cov = jnp.matmul(latent_sqrt, latent_sqrt.T)
        latent_dist = tfd.MultivariateNormalFullCovariance(latent_mean, latent_cov)

        # Perform checks specific to the given likelihood
        self._test_call_check(likelihood, latent_mean, latent_cov, latent_dist)

    @staticmethod
    def _test_call_check(likelihood, latent_mean, latent_cov, latent_dist):
        """Specific to each likelihood."""
        raise NotImplementedError


def prod(inp):
    return [
        dict(zip(inp.keys(), values, strict=True)) for values in product(*inp.values())
    ]


class TestGaussian(BaseTestLikelihood):
    likelihood = Gaussian
    fields = prod({"obs_stddev": [0.1, 0.5, 1.0]})
    params = {"test_initialisation": fields, "test_call": fields}
    static_fields = ["num_datapoints"]

    @staticmethod
    def _test_call_check(likelihood: Gaussian, latent_mean, latent_cov, latent_dist):
        # Test call method.
        pred_dist = likelihood(latent_dist)

        # Check that the distribution is a MultivariateNormalFullCovariance.
        assert isinstance(pred_dist, tfd.MultivariateNormalFullCovariance)

        # Check predictive mean and variance.
        assert (pred_dist.mean() == latent_mean).all()
        noise_matrix = jnp.eye(likelihood.num_datapoints) * likelihood.obs_stddev**2
        assert np.allclose(
            pred_dist.scale_tril, jnp.linalg.cholesky(latent_cov + noise_matrix)
        )


class TestBernoulli(BaseTestLikelihood):
    likelihood = Bernoulli
    fields = prod({})
    params = {"test_initialisation": fields, "test_call": fields}
    static_fields = ["num_datapoints"]

    @staticmethod
    def _test_call_check(
        likelihood: AbstractLikelihood, latent_mean, latent_cov, latent_dist
    ):
        # Test call method.
        pred_dist = likelihood(latent_dist)

        # Check that the distribution is a Bernoulli.
        assert isinstance(pred_dist, tfd.Bernoulli)

        # Check predictive mean and variance.

        p = inv_probit(latent_mean / jnp.sqrt(1.0 + jnp.diagonal(latent_cov)))
        assert (pred_dist.mean() == p).all()
        assert (pred_dist.variance() == p * (1.0 - p)).all()


class TestPoisson(BaseTestLikelihood):
    likelihood = Poisson
    fields = prod({})
    params = {"test_initialisation": fields, "test_call": fields}
    static_fields = ["num_datapoints"]

    @staticmethod
    def _test_call_check(
        likelihood: AbstractLikelihood, latent_mean, latent_cov, latent_dist
    ):
        # Test call method.
        pred_dist = likelihood(latent_dist)

        # Check that the distribution is a Poisson.
        assert isinstance(pred_dist, tfd.Poisson)

        # Check predictive mean and variance.
        rate = jnp.exp(latent_mean)
        assert (pred_dist.mean() == rate).all()


class TestAbstract(BaseTestLikelihood):
    class DummyLikelihood(AbstractLikelihood):
        def predict(self, dist: tfd.Distribution) -> tfd.Distribution:
            return tfd.Normal(0.0, 1.0)

        def link_function(self, f: Float[Array, "N 1"]) -> Float[Array, "N 1"]:
            return tfd.MultivariateNormalDiag(loc=f)

    likelihood = DummyLikelihood
    fields = prod({})
    params = {"test_initialisation": fields, "test_call": fields}
    static_fields = ["num_datapoints"]

    @staticmethod
    def _test_call_check(
        likelihood: AbstractLikelihood, latent_mean, latent_cov, latent_dist
    ):
        pred_dist = likelihood(latent_dist)
        assert isinstance(pred_dist, tfd.Normal)
