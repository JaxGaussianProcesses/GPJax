from chex import dataclass
from .kernel import Kernel
from .mean_functions import MeanFunction, Zero
from .likelihoods import Likelihood, Gaussian
from typing import Optional


@dataclass
class Prior:
    kernel: Kernel
    mean_function: Optional[MeanFunction] = Zero()
    name: Optional[str] = "Prior"

    def __mul__(self, other: Likelihood):
        return ConjugatePosterior(prior=self, likelihood=other)


@dataclass
class Posterior:
    prior: Prior
    likelihood: Likelihood
    name: Optional[str] = 'Posterior'


@dataclass
class ConjugatePosterior:
    prior: Prior
    likelihood: Gaussian
    name: Optional[str] = 'ConjugatePosterior'


@dataclass
class NonconjugatePosterior:
    prior: Prior
    likelihood: Gaussian
    name: Optional[str] = 'ConjugatePosterior'


