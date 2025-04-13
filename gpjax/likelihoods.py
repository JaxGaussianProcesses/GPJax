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

import beartype.typing as tp
from flax import nnx
from jax import vmap
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Float
import numpyro.distributions as npd

from gpjax.distributions import GaussianDistribution
from gpjax.integrators import (
    AbstractIntegrator,
    AnalyticalGaussianIntegrator,
    GHQuadratureIntegrator,
)
from gpjax.parameters import (
    PositiveReal,
    Static,
)
from gpjax.typing import (
    Array,
    ScalarFloat,
)


class AbstractLikelihood(nnx.Module):
    r"""Abstract base class for likelihoods.

    All likelihoods must inherit from this class and implement the `predict` and
    `link_function` methods.
    """

    def __init__(
        self,
        num_datapoints: int,
        integrator: AbstractIntegrator = GHQuadratureIntegrator(),
    ):
        """Initializes the likelihood.

        Args:
            num_datapoints (int): the number of data points.
            integrator (AbstractIntegrator): The integrator to be used for computing expected log
                likelihoods. Must be an instance of `AbstractIntegrator`.
        """
        self.num_datapoints = num_datapoints
        self.integrator = integrator

    def __call__(self, *args: tp.Any, **kwargs: tp.Any) -> npd.Distribution:
        r"""Evaluate the likelihood function at a given predictive distribution.

        Args:
            *args (Any): Arguments to be passed to the likelihood's `predict` method.
            **kwargs (Any): Keyword arguments to be passed to the likelihood's
                `predict` method.

        Returns:
            The predictive distribution.
        """
        return self.predict(*args, **kwargs)

    @abc.abstractmethod
    def predict(self, *args: tp.Any, **kwargs: tp.Any) -> npd.Distribution:
        r"""Evaluate the likelihood function at a given predictive distribution.

        Args:
            *args (Any): Arguments to be passed to the likelihood's `predict` method.
            **kwargs (Any): Keyword arguments to be passed to the likelihood's
                `predict` method.

        Returns:
            npd.Distribution: The predictive distribution.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def link_function(self, f: Float[Array, "..."]) -> npd.Distribution:
        r"""Return the link function of the likelihood function.

        Args:
            f (Float[Array, "..."]): the latent Gaussian process values.

        Returns:
            npd.Distribution: The distribution of observations, y, given values of the
                Gaussian process, f.
        """
        raise NotImplementedError

    def expected_log_likelihood(
        self,
        y: Float[Array, "N D"],
        mean: Float[Array, "N D"],
        variance: Float[Array, "N D"],
    ) -> Float[Array, " N"]:
        r"""Compute the expected log likelihood.

        For a variational distribution $q(f)\sim\mathcal{N}(m, s)$ and a likelihood
        $p(y|f)$, compute the expected log likelihood:
        ```math
        \mathbb{E}_{q(f)}\left[\log p(y|f)\right]
        ```

        Args:
            y (Float[Array, 'N D']): The observed response variable.
            mean (Float[Array, 'N D']): The variational mean.
            variance (Float[Array, 'N D']): The variational variance.

        Returns:
            ScalarFloat: The expected log likelihood.
        """
        log_prob = vmap(lambda f, y: self.link_function(f).log_prob(y))
        return self.integrator(
            fun=log_prob, y=y, mean=mean, variance=variance, likelihood=self
        )


class Gaussian(AbstractLikelihood):
    r"""Gaussian likelihood object."""

    def __init__(
        self,
        num_datapoints: int,
        obs_stddev: tp.Union[
            ScalarFloat, Float[Array, "#N"], PositiveReal, Static
        ] = 1.0,
        integrator: AbstractIntegrator = AnalyticalGaussianIntegrator(),
    ):
        r"""Initializes the Gaussian likelihood.

        Args:
            num_datapoints (int): the number of data points.
            obs_stddev (Union[ScalarFloat, Float[Array, "#N"]]): the standard deviation
                of the Gaussian observation noise.
            integrator (AbstractIntegrator): The integrator to be used for computing expected log
                likelihoods. Must be an instance of `AbstractIntegrator`. For the Gaussian likelihood, this defaults to
                the `AnalyticalGaussianIntegrator`, as the expected log likelihood can be computed analytically.
        """
        if not isinstance(obs_stddev, (PositiveReal, Static)):
            obs_stddev = PositiveReal(jnp.asarray(obs_stddev))
        self.obs_stddev = obs_stddev

        super().__init__(num_datapoints, integrator)

    def link_function(self, f: Float[Array, "..."]) -> npd.Normal:
        r"""The link function of the Gaussian likelihood.

        Args:
            f (Float[Array, "..."]): Function values.

        Returns:
            npd.Normal: The likelihood function.
        """
        return npd.Normal(loc=f, scale=self.obs_stddev.value.astype(f.dtype))

    def predict(
        self, dist: tp.Union[npd.MultivariateNormal, GaussianDistribution]
    ) -> npd.MultivariateNormal:
        r"""Evaluate the Gaussian likelihood.

        Evaluate the Gaussian likelihood function at a given predictive
        distribution. Computationally, this is equivalent to summing the
        observation noise term to the diagonal elements of the predictive
        distribution's covariance matrix.

        Args:
            dist (npd.Distribution): The Gaussian process posterior,
                evaluated at a finite set of test points.

        Returns:
            npd.Distribution: The predictive distribution.
        """
        n_data = dist.event_shape[0]
        cov = dist.covariance_matrix
        noisy_cov = cov.at[jnp.diag_indices(n_data)].add(self.obs_stddev.value**2)

        return npd.MultivariateNormal(dist.mean, noisy_cov)


class Bernoulli(AbstractLikelihood):
    def link_function(self, f: Float[Array, "..."]) -> npd.BernoulliProbs:
        r"""The probit link function of the Bernoulli likelihood.

        Args:
            f (Float[Array, "..."]): Function values.

        Returns:
            npd.Bernoulli: The likelihood function.
        """
        return npd.Bernoulli(probs=inv_probit(f))

    def predict(
        self, dist: tp.Union[npd.MultivariateNormal, GaussianDistribution]
    ) -> npd.BernoulliProbs:
        r"""Evaluate the pointwise predictive distribution.

        Evaluate the pointwise predictive distribution, given a Gaussian
        process posterior and likelihood parameters.

        Args:
            dist ([npd.MultivariateNormal, GaussianDistribution].): The Gaussian
                process posterior, evaluated at a finite set of test points.

        Returns:
            npd.Bernoulli: The pointwise predictive distribution.
        """
        variance = jnp.diag(dist.covariance_matrix)
        mean = dist.mean.ravel()
        return self.link_function(mean / jnp.sqrt(1.0 + variance))


class Poisson(AbstractLikelihood):
    def link_function(self, f: Float[Array, "..."]) -> npd.Poisson:
        r"""The link function of the Poisson likelihood.

        Args:
            f (Float[Array, "..."]): Function values.

        Returns:
            npd.Poisson: The likelihood function.
        """
        return npd.Poisson(rate=jnp.exp(f))

    def predict(
        self, dist: tp.Union[npd.MultivariateNormal, GaussianDistribution]
    ) -> npd.Poisson:
        r"""Evaluate the pointwise predictive distribution.

        Evaluate the pointwise predictive distribution, given a Gaussian
        process posterior and likelihood parameters.

        Args:
            dist (tp.Union[npd.MultivariateNormal, GaussianDistribution]): The Gaussian
                process posterior, evaluated at a finite set of test points.

        Returns:
            npd.Poisson: The pointwise predictive distribution.
        """
        return self.link_function(dist.mean)


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


NonGaussian = tp.Union[Poisson, Bernoulli]

__all__ = [
    "AbstractLikelihood",
    "NonGaussian",
    "Gaussian",
    "Bernoulli",
    "Poisson",
    "inv_probit",
]
