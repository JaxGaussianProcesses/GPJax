import abc
from typing import Any, Callable, Optional

import distrax as dx
import jax.numpy as jnp
import jax.scipy as jsp
from chex import dataclass

from .types import Array
from .utils import I


@dataclass
class AbstractLikelihood:
    """Abstract base class for likelihoods."""

    num_datapoints: int  # The number of datapoints that the likelihood factorises over
    name: Optional[str] = "Likelihood"

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Evaluate the likelihood function at a given predictive distribution."""
        return self.predict(*args, **kwargs)

    @abc.abstractmethod
    def predict(self, *args: Any, **kwargs: Any) -> Any:
        """Evaluate the likelihood function at a given predictive distribution."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def params(self) -> dict:
        """Return the parameters of the likelihood function."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def link_function(self) -> Callable:
        """Return the link function of the likelihood function."""
        raise NotImplementedError


@dataclass
class Conjugate:
    """Conjugate likelihood."""


@dataclass
class NonConjugate:
    """Conjugate likelihood."""


@dataclass
class Gaussian(AbstractLikelihood, Conjugate):
    """Gaussian likelihood object."""

    name: Optional[str] = "Gaussian"

    @property
    def params(self) -> dict:
        """Return the variance parameter of the likelihood function."""
        return {"obs_noise": jnp.array([1.0])}

    @property
    def link_function(self) -> Callable:
        """Return the link function of the Gaussian likelihood. Here, this is simply the identity function, but we include it for completeness.

        Returns:
            Callable: A link function that maps the predictive distribution to the likelihood function.
        """

        def link_fn(x, params: dict) -> dx.Distribution:
            return dx.Normal(loc=x, scale=params["obs_noise"])

        return link_fn

    def predict(self, dist: dx.Distribution, params: dict) -> dx.Distribution:
        """Evaluate the Gaussian likelihood function at a given predictive distribution. Computationally, this is equivalent to summing the observation noise term to the diagonal elements of the predictive distribution's covariance matrix.."""
        n_data = dist.event_shape[0]
        noisy_cov = dist.covariance() + I(n_data) * params["likelihood"]["obs_noise"]
        return dx.MultivariateNormalFullCovariance(dist.mean(), noisy_cov)


@dataclass
class Bernoulli(AbstractLikelihood, NonConjugate):
    name: Optional[str] = "Bernoulli"

    @property
    def params(self) -> dict:
        return {}

    @property
    def link_function(self) -> Callable:
        def link_fn(x, params: dict) -> dx.Distribution:
            return dx.Bernoulli(probs=inv_probit(x))

        return link_fn

    @property
    def predictive_moment_fn(self) -> Callable:
        def moment_fn(mean: Array, variance: Array):
            rv = self.link_function(mean / jnp.sqrt(1 + variance), self.params)
            return rv

        return moment_fn

    def predict(self, dist: dx.Distribution, params: dict) -> Any:
        variance = jnp.diag(dist.covariance())
        mean = dist.mean()
        return self.predictive_moment_fn(mean.ravel(), variance)


def inv_probit(x):
    jitter = 1e-3  # ensures output is strictly between 0 and 1
    return 0.5 * (1.0 + jsp.special.erf(x / jnp.sqrt(2.0))) * (1 - 2 * jitter) + jitter


NonConjugateLikelihoods = [Bernoulli]
NonConjugateLikelihoodType = Bernoulli  # Union[Bernoulli]
