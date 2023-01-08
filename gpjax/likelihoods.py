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
from jaxlinop.utils import to_dense
from jaxutils import PyTree

import distrax as dx
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array, Float

from jax.random import KeyArray

import deprecation


class AbstractLikelihood(PyTree):
    """Abstract base class for likelihoods."""

    def __init__(self, num_datapoints: int, name: Optional[str] = None):
        """Initialise the likelihood.

        Args:
            num_datapoints (int): The number of datapoints that the likelihood factorises over.
            name (Optional[str]): The name of the likelihood. Defaults to None.
        """
        self.num_datapoints = num_datapoints
        self.name = name

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

    @abc.abstractmethod
    def init_params(self, key: KeyArray) -> Dict:
        """Return the parameters of the likelihood function.

        Args:
            key (KeyArray): A PRNG key.

        Returns:
            Dict: The parameters of the likelihood function.
        """
        raise NotImplementedError

    @deprecation.deprecated(
        deprecated_in="0.5.7",
        removed_in="0.6.0",
        details="Use the ``init_params`` method for parameter initialisation.",
    )
    def _initialise_params(self, key: KeyArray) -> Dict:
        """Deprecated method for initialising the GP's parameters. Succeded by ``init_params``."""
        return self.init_params(key)

    @property
    @abc.abstractmethod
    def link_function(self) -> Callable:
        """Return the link function of the likelihood function.

        Returns:
            Callable: The link function of the likelihood function.
        """
        raise NotImplementedError


class Conjugate:
    """An abstract class for conjugate likelihoods with respect to a Gaussain process prior."""


class NonConjugate:
    """An abstract class for non-conjugate likelihoods with respect to a Gaussain process prior."""


# TODO: revamp this with covariance operators.


class Gaussian(AbstractLikelihood, Conjugate):
    """Gaussian likelihood object."""

    def __init__(self, num_datapoints: int, name: Optional[str] = "Gaussian"):
        """Initialise the Gaussian likelihood.

        Args:
            num_datapoints (int): The number of datapoints that the likelihood factorises over.
            name (Optional[str]): The name of the likelihood. Defaults to "Gaussian".
        """
        super().__init__(num_datapoints, name)

    def init_params(self, key: KeyArray) -> Dict:
        """Return the variance parameter of the likelihood function.

        Args:
            key (KeyArray): A PRNG key.

        Returns:
            Dict: The parameters of the likelihood function.
        """
        return {"obs_noise": jnp.array([1.0])}

    @property
    def link_function(self) -> Callable[[Dict, Float[Array, "N 1"]], dx.Distribution]:
        """Return the link function of the Gaussian likelihood. Here, this is
        simply the identity function, but we include it for completeness.

        Returns:
            Callable[[Dict, Float[Array, "N 1"]], dx.Distribution]: A link
            function that maps the predictive distribution to the likelihood function.
        """

        def link_fn(params: Dict, f: Float[Array, "N 1"]) -> dx.Normal:
            """The link function of the Gaussian likelihood.

            Args:
                params (Dict): The parameters of the likelihood function.
                f (Float[Array, "N 1"]): Function values.

            Returns:
                dx.Normal: The likelihood function.
            """
            return dx.Normal(loc=f, scale=params["obs_noise"])

        return link_fn

    def predict(self, params: Dict, dist: dx.MultivariateNormalTri) -> dx.Distribution:
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
        noisy_cov = cov.at[jnp.diag_indices(n_data)].add(
            params["likelihood"]["obs_noise"]
        )

        return dx.MultivariateNormalFullCovariance(dist.mean(), noisy_cov)


class Bernoulli(AbstractLikelihood, NonConjugate):
    def __init__(self, num_datapoints: int, name: Optional[str] = "Bernoulli"):
        """Initialise the Bernoulli likelihood.

        Args:
            num_datapoints (int): The number of datapoints that the likelihood factorises over.
            name (Optional[str]): The name of the likelihood. Defaults to "Bernoulli".
        """
        super().__init__(num_datapoints, name)

    def init_params(self, key: KeyArray) -> Dict:
        """Initialise the parameter set of a Bernoulli likelihood.

        Args:
            key (KeyArray): A PRNG key.

        Returns:
            Dict: The parameters of the likelihood function (empty for the Bernoulli likelihood).
        """
        return {}

    @property
    def link_function(self) -> Callable[[Dict, Float[Array, "N 1"]], dx.Distribution]:
        """Return the probit link function of the Bernoulli likelihood.

        Returns:
            Callable[[Dict, Float[Array, "N 1"]], dx.Distribution]: A probit link
                function that maps the predictive distribution to the likelihood function.
        """

        def link_fn(params: Dict, f: Float[Array, "N 1"]) -> dx.Distribution:
            """The probit link function of the Bernoulli likelihood.

            Args:
                params (Dict): The parameters of the likelihood function.
                f (Float[Array, "N 1"]): Function values.

            Returns:
                dx.Distribution: The likelihood function.
            """
            return dx.Bernoulli(probs=inv_probit(f))

        return link_fn

    @property
    def predictive_moment_fn(
        self,
    ) -> Callable[[Dict, Float[Array, "N 1"]], Float[Array, "N 1"]]:
        """Instantiate the predictive moment function of the Bernoulli likelihood
        that is parameterised by a probit link function.

        Returns:
            Callable: A callable object that accepts a mean and variance term
                from which the predictive random variable is computed.
        """

        def moment_fn(
            params: Dict,
            mean: Float[Array, "N 1"],
            variance: Float[Array, "N 1"],
        ):
            """The predictive moment function of the Bernoulli likelihood.

            Args:
                params (Dict): The parameters of the likelihood function.
                mean (Float[Array, "N 1"]): The mean of the latent function values.
                variance (Float[Array, "N 1"]): The diagonal variance of the latent function values.

            Returns:
                Float[Array, "N 1"]: The pointwise predictive distribution.
            """
            rv = self.link_function(params, mean / jnp.sqrt(1.0 + variance))
            return rv

        return moment_fn

    def predict(self, params: Dict, dist: dx.Distribution) -> dx.Distribution:
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
        return self.predictive_moment_fn(params, mean, variance)


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
