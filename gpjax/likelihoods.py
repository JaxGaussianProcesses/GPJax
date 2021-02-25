from chex import dataclass
from typing import Optional, Union, Callable
from multipledispatch import dispatch
import jax.numpy as jnp
from tensorflow_probability.substrates.jax import distributions as tfd


@dataclass
class Likelihood:
    name: Optional[str] = 'Likelihood'


@dataclass
class Gaussian:
    name: Optional[str] = 'Gaussian'


@dispatch(Gaussian)
def initialise(likelihood: Gaussian) -> dict:
    return {'obs_noise': jnp.array([1.0])}


@dataclass
class Bernoulli:
    name: Optional[str] = 'Bernoulli'


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



@dataclass
class Poisson:
    name: Optional[str] = 'Poisson'


NonConjugateLikelihoods = (Bernoulli, Poisson)
NonConjugateLikelihoodType = Union[Bernoulli, Poisson]