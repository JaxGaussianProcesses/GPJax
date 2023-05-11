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
from dataclasses import dataclass

from beartype.typing import (
    Any,
    Union,
)
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Float
import tensorflow_probability.substrates.jax as tfp

from gpjax.base import (
    Module,
    param_field,
    static_field,
)
from gpjax.gaussian_distribution import GaussianDistribution
from gpjax.linops.utils import to_dense
from gpjax.typing import (
    Array,
    ScalarFloat,
)

tfb = tfp.bijectors
tfd = tfp.distributions


@dataclass
class AbstractLikelihood(Module):
    r"""Abstract base class for likelihoods."""

    num_datapoints: int = static_field()

    def __call__(self, *args: Any, **kwargs: Any) -> tfd.Distribution:
        r"""Evaluate the likelihood function at a given predictive distribution.

        Args:
            *args (Any): Arguments to be passed to the likelihood's `predict` method.
            **kwargs (Any): Keyword arguments to be passed to the likelihood's `predict` method.

        Returns
        -------
            tfd.Distribution: The predictive distribution.
        """
        return self.predict(*args, **kwargs)

    @abc.abstractmethod
    def predict(self, *args: Any, **kwargs: Any) -> tfd.Distribution:
        r"""Evaluate the likelihood function at a given predictive distribution.

        Args:
            *args (Any): Arguments to be passed to the likelihood's `predict` method.
            **kwargs (Any): Keyword arguments to be passed to the likelihood's `predict` method.

        Returns
        -------
            tfd.Distribution: The predictive distribution.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def link_function(self, f: Float[Array, "..."]) -> tfd.Distribution:
        r"""Return the link function of the likelihood function.

        Returns
        -------
            tfd.Distribution: The distribution of observations, y, given values of the Gaussian process, f.
        """
        raise NotImplementedError


@dataclass
class Gaussian(AbstractLikelihood):
    r"""Gaussian likelihood object."""

    obs_noise: Union[ScalarFloat, Float[Array, "#N"]] = param_field(
        jnp.array(1.0), bijector=tfb.Softplus()
    )

    def link_function(self, f: Float[Array, "..."]) -> tfd.Normal:
        r"""The link function of the Gaussian likelihood.

        Args:
            f (Float[Array, "..."]): Function values.

        Returns
        -------
            tfd.Normal: The likelihood function.
        """
        return tfd.Normal(loc=f, scale=self.obs_noise.astype(f.dtype))

    def predict(
        self, dist: Union[tfd.MultivariateNormalTriL, GaussianDistribution]
    ) -> tfd.MultivariateNormalFullCovariance:
        r"""Evaluate the Gaussian likelihood.

        Evaluate the Gaussian likelihood function at a given predictive
        distribution. Computationally, this is equivalent to summing the
        observation noise term to the diagonal elements of the predictive
        distribution's covariance matrix.

        Args:
            dist (tfd.Distribution): The Gaussian process posterior,
                evaluated at a finite set of test points.

        Returns
        -------
            tfd.Distribution: The predictive distribution.
        """
        n_data = dist.event_shape[0]
        cov = to_dense(dist.covariance())
        noisy_cov = cov.at[jnp.diag_indices(n_data)].add(self.obs_noise)

        return tfd.MultivariateNormalFullCovariance(dist.mean(), noisy_cov)


@dataclass
class Bernoulli(AbstractLikelihood):
    def link_function(self, f: Float[Array, "..."]) -> tfd.Distribution:
        r"""The probit link function of the Bernoulli likelihood.

        Args:
            f (Float[Array, "..."]): Function values.

        Returns
        -------
            tfd.Distribution: The likelihood function.
        """
        return tfd.Bernoulli(probs=inv_probit(f))

    def predict(self, dist: tfd.Distribution) -> tfd.Distribution:
        r"""Evaluate the pointwise predictive distribution.

        Evaluate the pointwise predictive distribution, given a Gaussian
        process posterior and likelihood parameters.

        Args:
            dist (tfd.Distribution): The Gaussian process posterior, evaluated
                at a finite set of test points.

        Returns
        -------
            tfd.Distribution: The pointwise predictive distribution.
        """
        variance = jnp.diag(dist.covariance())
        mean = dist.mean().ravel()
        return self.link_function(mean / jnp.sqrt(1.0 + variance))


@dataclass
class Poisson(AbstractLikelihood):
    def link_function(self, f: Float[Array, "..."]) -> tfd.Distribution:
        r"""The link function of the Poisson likelihood.

        Args:
            f (Float[Array, "..."]): Function values.

        Returns:
            tfd.Distribution: The likelihood function.
        """
        return tfd.Poisson(rate=jnp.exp(f))

    def predict(self, dist: tfd.Distribution) -> tfd.Distribution:
        r"""Evaluate the pointwise predictive distribution.

        Evaluate the pointwise predictive distribution, given a Gaussian
        process posterior and likelihood parameters.

        Args:
            dist (tfd.Distribution): The Gaussian process posterior, evaluated
                at a finite set of test points.

        Returns:
            tfd.Distribution: The pointwise predictive distribution.
        """
        return self.link_function(dist.mean())


def inv_probit(x: Float[Array, " *N"]) -> Float[Array, " *N"]:
    r"""Compute the inverse probit function.

    Args:
        x (Float[Array, "*N"]): A vector of values.

    Returns
    -------
        Float[Array, "*N"]: The inverse probit of the input vector.
    """
    jitter = 1e-3  # To ensure output is in interval (0, 1).
    return 0.5 * (1.0 + jsp.special.erf(x / jnp.sqrt(2.0))) * (1 - 2 * jitter) + jitter


__all__ = [
    "AbstractLikelihood",
    "Gaussian",
    "Bernoulli",
    "Poisson",
    "inv_probit",
]
