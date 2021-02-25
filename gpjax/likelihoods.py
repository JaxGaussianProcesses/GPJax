from chex import dataclass
from typing import Optional, Union
from multipledispatch import dispatch
import jax.numpy as jnp


@dataclass
class Likelihood:
    name: Optional[str] = 'Likelihood'


@dataclass
class Gaussian:
    name: Optional[str] = 'Gaussian'


@dispatch(Gaussian)
def initialise(likelihood: Gaussian):
    return {'obs_noise': jnp.array([1.0])}


@dataclass
class Bernoulli:
    name: Optional[str] = 'Bernoulli'


@dataclass
class Poisson:
    name: Optional[str] = 'Poisson'


NonConjugateLikelihoods = (Bernoulli, Poisson)
NonConjugateLikelihoodType = Union[Bernoulli, Poisson]