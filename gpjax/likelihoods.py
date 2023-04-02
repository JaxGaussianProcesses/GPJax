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

import abc
from typing import Any
from .linops.utils import to_dense

import distrax as dx
import tensorflow_probability.substrates.jax.bijectors as tfb
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array, Float
from simple_pytree import static_field

from dataclasses import dataclass
from .base import Module, param_field

@dataclass
class AbstractLikelihood(Module):
    """Abstract base class for likelihoods."""
    num_datapoints: int = static_field()


    def __call__(self, *args: Any, **kwargs: Any) -> dx.Distribution:
        """Evaluate the likelihood function at a given predictive distribution.

        Args:
            *args (Any): Arguments to be passed to the likelihood's `predict` method.
            **kwargs (Any): Keyword arguments to be passed to the likelihood's `predict` method.

        Returns:
            dx.Distribution: The predictive distribution.
        """
        return self.predict(*args, **kwargs)

    @abc.abstractmethod
    def predict(self, *args: Any, **kwargs: Any) -> dx.Distribution:
        """Evaluate the likelihood function at a given predictive distribution.

        Args:
            *args (Any): Arguments to be passed to the likelihood's `predict` method.
            **kwargs (Any): Keyword arguments to be passed to the likelihood's `predict` method.

        Returns:
            dx.Distribution: The predictive distribution.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def link_function(self) -> dx.Distribution:
        """Return the link function of the likelihood function.

        Returns:
            dx.Distribution: The distribution of observations, y, given values of the Gaussian process, f.
        """
        raise NotImplementedError


@dataclass
class Gaussian(AbstractLikelihood):
    """Gaussian likelihood object."""
    obs_noise: Float[Array, "1"] = param_field(jnp.array([1.0]), bijector=tfb.Softplus())

    def link_function(self, f: Float[Array, "N 1"]) -> dx.Normal:
        """The link function of the Gaussian likelihood.

        Args:
            params (Dict): The parameters of the likelihood function.
            f (Float[Array, "N 1"]): Function values.

        Returns:
            dx.Normal: The likelihood function.
        """
        return dx.Normal(loc=f, scale=self.obs_noise)

    def predict(self, dist: dx.MultivariateNormalTri) -> dx.Distribution:
        """
        Evaluate the Gaussian likelihood function at a given predictive
        distribution. Computationally, this is equivalent to summing the
        observation noise term to the diagonal elements of the predictive
        distribution's covariance matrix.

        Args:
            params (Dict): The parameters of the likelihood function.
            dist (dx.Distribution): The Gaussian process posterior,
                evaluated at a finite set of test points.

        Returns:
            dx.Distribution: The predictive distribution.
        """
        n_data = dist.event_shape[0]
        cov = to_dense(dist.covariance())
        noisy_cov = cov.at[jnp.diag_indices(n_data)].add(self.obs_noise)

        return dx.MultivariateNormalFullCovariance(dist.mean(), noisy_cov)


@dataclass
class Bernoulli(AbstractLikelihood):

    def link_function(self, f: Float[Array, "N 1"]) -> dx.Distribution:
        """The probit link function of the Bernoulli likelihood.

        Args:
            f (Float[Array, "N 1"]): Function values.

        Returns:
            dx.Distribution: The likelihood function.
        """
        return dx.Bernoulli(probs=inv_probit(f))

    def predict(self, dist: dx.Distribution) -> dx.Distribution:
        """Evaluate the pointwise predictive distribution, given a Gaussian
        process posterior and likelihood parameters.

        Args:
            params (Dict): The parameters of the likelihood function.
            dist (dx.Distribution): The Gaussian process posterior, evaluated
                at a finite set of test points.

        Returns:
            dx.Distribution: The pointwise predictive distribution.
        """
        variance = jnp.diag(dist.covariance())
        mean = dist.mean().ravel()
        return  self.link_function(mean / jnp.sqrt(1.0 + variance))


def inv_probit(x: Float[Array, "N 1"]) -> Float[Array, "N 1"]:
    """Compute the inverse probit function.

    Args:
        x (Float[Array, "N 1"]): A vector of values.

    Returns:
        Float[Array, "N 1"]: The inverse probit of the input vector.
    """
    jitter = 1e-3  # To ensure output is in interval (0, 1).
    return 0.5 * (1.0 + jsp.special.erf(x / jnp.sqrt(2.0))) * (1 - 2 * jitter) + jitter


__all__ = [
    "AbstractLikelihood",
    "Conjugate",
    "NonConjugate",
    "Gaussian",
    "Bernoulli",
    "inv_probit",
]
