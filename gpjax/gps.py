from typing import Optional

from chex import dataclass
from multipledispatch import dispatch

from .kernels import Kernel
from .likelihoods import (Gaussian, Likelihood, NonConjugateLikelihoods,
                          NonConjugateLikelihoodType)
from .mean_functions import MeanFunction, Zero


@dataclass
class Prior:
    kernel: Kernel
    mean_function: Optional[MeanFunction] = Zero()
    name: Optional[str] = "Prior"

    @dispatch(Gaussian)
    def __mul__(self, other: Gaussian):
        return ConjugatePosterior(prior=self, likelihood=other)

    @dispatch(NonConjugateLikelihoods)
    def __mul__(self, other: NonConjugateLikelihoodType):
        return NonConjugatePosterior(prior=self, likelihood=other)


@dataclass
class Posterior:
    prior: Prior
    likelihood: Likelihood
    name: Optional[str] = "Posterior"


@dataclass
class ConjugatePosterior:
    prior: Prior
    likelihood: Gaussian
    name: Optional[str] = "ConjugatePosterior"


@dataclass
class NonConjugatePosterior:
    prior: Prior
    likelihood: NonConjugateLikelihoodType
    name: Optional[str] = "ConjugatePosterior"
