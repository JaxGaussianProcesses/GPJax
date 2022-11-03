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
from typing import Any, Callable, Dict, Optional

import jax.numpy as jnp
import jax.scipy as jsp
import numpyro.distributions as npd
from chex import dataclass
from jaxtyping import Array, Float

from .types import PRNGKeyType


@dataclass
class AbstractLikelihood:
    """Abstract base class for likelihoods."""

    num_datapoints: int  # The number of datapoints that the likelihood factorises over.
    name: Optional[str] = "Likelihood"

    def __call__(self, *args: Any, **kwargs: Any) -> npd.Distribution:
        """Evaluate the likelihood function at a given predictive distribution.

        Args:
            *args (Any): Arguments to be passed to the likelihood's `predict` method.
            **kwargs (Any): Keyword arguments to be passed to the likelihood's `predict` method.

        Returns:
            dx.Distribution: The predictive distribution.
        """
        return self.predict(*args, **kwargs)

    @abc.abstractmethod
    def predict(self, *args: Any, **kwargs: Any) -> npd.Distribution:
        """Evaluate the likelihood function at a given predictive distribution.

        Args:
            *args (Any): Arguments to be passed to the likelihood's `predict` method.
            **kwargs (Any): Keyword arguments to be passed to the likelihood's `predict` method.

        Returns:
            dx.Distribution: The predictive distribution.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _initialise_params(self, key: PRNGKeyType) -> Dict:
        """Return the parameters of the likelihood function.

        Args:
            key (PRNGKeyType): A PRNG key.

        Returns:
            Dict: The parameters of the likelihood function.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def link_function(self) -> Callable:
        """Return the link function of the likelihood function.

        Returns:
            Callable: The link function of the likelihood function.
        """
        raise NotImplementedError


@dataclass
class Conjugate:
    """An abstract class for conjugate likelihoods with respect to a Gaussain process prior."""


@dataclass
class NonConjugate:
    """An abstract class for non-conjugate likelihoods with respect to a Gaussain process prior."""


# TODO: revamp this will covariance operators.
@dataclass
class Gaussian(AbstractLikelihood, Conjugate):
    """Gaussian likelihood object."""

    name: Optional[str] = "Gaussian"

    def _initialise_params(self, key: PRNGKeyType) -> Dict:
        """Return the variance parameter of the likelihood function.

        Args:
            key (PRNGKeyType): A PRNG key.

        Returns:
            Dict: The parameters of the likelihood function.
        """
        return {"obs_noise": jnp.array([1.0])}

    @property
    def link_function(self) -> Callable:
        """Return the link function of the Gaussian likelihood. Here, this is simply the identity function, but we include it for completeness.

        Returns:
            Callable: A link function that maps the predictive distribution to the likelihood function.
        """

        def link_fn(x, params: Dict) -> npd.Distribution:
            return npd.Normal(loc=x, scale=params["obs_noise"])

        return link_fn

    def predict(self, dist: npd.Distribution, params: Dict) -> npd.Distribution:
        """Evaluate the Gaussian likelihood function at a given predictive distribution. Computationally, this is equivalent to summing the observation noise term to the diagonal elements of the predictive distribution's covariance matrix.

        Args:
            dist (numpyro.distributions.Distribution): The Gaussian process posterior, evaluated at a finite set of test points.
            params (Dict): The parameters of the likelihood function.

        Returns:
            numpyro.distributions.Distribution: The predictive distribution.
        """
        n_data = dist.event_shape[0]
<<<<<<< HEAD
        cov = dist.covariance()
        noisy_cov = cov.at[jnp.diag_indices(n_data)].add(
            params["likelihood"]["obs_noise"]
        )

        return dx.MultivariateNormalFullCovariance(dist.mean(), noisy_cov)
=======
        noisy_cov = dist.covariance_matrix + I(n_data) * params["likelihood"]["obs_noise"]
        return npd.MultivariateNormal(dist.mean, noisy_cov)
>>>>>>> v0.5_update


@dataclass
class Bernoulli(AbstractLikelihood, NonConjugate):
    name: Optional[str] = "Bernoulli"

    def _initialise_params(self, key: PRNGKeyType) -> Dict:
        """Initialise the parameter set of a Bernoulli likelihood.

        Args:
            key (PRNGKeyType): A PRNG key.

        Returns:
            Dict: The parameters of the likelihood function (empty for the Bernoulli likelihood).
        """
        return {}

    @property
    def link_function(self) -> Callable:
        """Return the probit link function of the Bernoulli likelihood.

        Returns:
            Callable: A probit link function that maps the predictive distribution to the likelihood function.
        """

        def link_fn(x, params: Dict) -> npd.Distribution:
            return npd.Bernoulli(probs=inv_probit(x))

        return link_fn

    @property
    def predictive_moment_fn(self) -> Callable:
        """Instantiate the predictive moment function of the Bernoulli likelihood that is parameterised by a probit link function.

        Returns:
            Callable: A callable object that accepts a mean and variance term from which the predictive random variable is computed.
        """

        def moment_fn(mean: Float[Array, "N D"], variance: Float[Array, "N D"], params: Dict):
            rv = self.link_function(mean / jnp.sqrt(1 + variance), params)
            return rv

        return moment_fn

    def predict(self, dist: npd.Distribution, params: Dict) -> npd.Distribution:
        """Evaluate the pointwise predictive distribution, given a Gaussian process posterior and likelihood parameters.

        Args:
            dist (npd.Distribution): The Gaussian process posterior, evaluated at a finite set of test points.
            params (Dict): The parameters of the likelihood function.

        Returns:
            npd.Distribution: The pointwise predictive distribution.
        """
        variance = jnp.diag(dist.covariance_matrix)
        mean = dist.mean
        return self.predictive_moment_fn(mean.ravel(), variance, params)


def inv_probit(x: Float[Array, "N 1"]) -> Float[Array, "N 1"]:
    """Compute the inverse probit function.

    Args:
        x (Float[Array, "N 1"]): A vector of values.

    Returns:
        Float[Array, "N 1"]: The inverse probit of the input vector.
    """
    jitter = 1e-3  # ensures output is strictly between 0 and 1
    return 0.5 * (1.0 + jsp.special.erf(x / jnp.sqrt(2.0))) * (1 - 2 * jitter) + jitter


__all__ = [
    "AbstractLikelihood",
    "Conjugate",
    "NonConjugate",
    "Gaussian",
    "Bernoulli",
    "inv_probit",
]
