import abc
from typing import Callable, Optional

import jax.numpy as jnp
from chex import dataclass
from tensorflow_probability.substrates.jax import distributions as tfd


@dataclass(repr=False)
class Likelihood:
    num_datapoints: int  # The number of datapoints that the likelihood factorises over
    name: Optional[str] = "Likelihood"

    def __repr__(self):
        return f"{self.name} likelihood function"

    @property
    @abc.abstractmethod
    def params(self) -> dict:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def link_function(self) -> Callable:
        raise NotImplementedError


@dataclass(repr=False)
class Gaussian(Likelihood):
    name: Optional[str] = "Gaussian"

    @property
    def params(self) -> dict:
        return {"obs_noise": jnp.array([1.0])}

    @property
    def link_function(self) -> Callable:
        def identity_fn(x):
            return x

        return identity_fn


@dataclass(repr=False)
class Bernoulli(Likelihood):
    name: Optional[str] = "Bernoulli"

    @property
    def params(self) -> dict:
        return {}

    @property
    def link_function(self) -> Callable:
        def link_fn(x):
            return tfd.ProbitBernoulli(x)

        return link_fn

    @property
    def predictive_moment_fn(self) -> Callable:
        def moment_fn(mean: jnp.DeviceArray, variance: jnp.DeviceArray):
            rv = self.link_function(mean / jnp.sqrt(1 + variance))
            return rv

        return moment_fn


NonConjugateLikelihoods = [Bernoulli]
NonConjugateLikelihoodType = Bernoulli  # Union[Bernoulli]
