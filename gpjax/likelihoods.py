from typing import Callable, Optional, Union

import jax.numpy as jnp
from chex import dataclass
from multipledispatch import dispatch
from tensorflow_probability.substrates.jax import distributions as tfd


@dataclass(repr=False)
class Likelihood:
    name: Optional[str] = "Likelihood"

    def __repr__(self):
        return f"{self.name} likelihood function"


@dataclass(repr=False)
class Gaussian(Likelihood):
    name: Optional[str] = "Gaussian"


@dispatch(Gaussian)
def initialise(likelihood: Gaussian) -> dict:
    return {"obs_noise": jnp.array([1.0])}


@dataclass(repr=False)
class Bernoulli(Likelihood):
    name: Optional[str] = "Bernoulli"


@dispatch(Bernoulli)
def initialise(likelihood: Bernoulli) -> dict:
    return {}


@dispatch(Bernoulli)
def link_function(likelihood: Bernoulli):
    return tfd.ProbitBernoulli


@dispatch(Bernoulli)
def predictive_moments(likelihood: Bernoulli) -> Callable:
    link = link_function(likelihood)

    def moments(mean: jnp.DeviceArray, variance: jnp.DeviceArray) -> tfd.Distribution:
        rv = link(mean / jnp.sqrt(1 + variance))
        return rv

    return moments


@dataclass(repr=False)
class Poisson(Likelihood):
    name: Optional[str] = "Poisson"


@dispatch(Poisson)
def initialise(likelihood: Poisson) -> dict:
    return {}


NonConjugateLikelihoods = (Bernoulli, Poisson)
NonConjugateLikelihoodType = Union[Bernoulli, Poisson]
