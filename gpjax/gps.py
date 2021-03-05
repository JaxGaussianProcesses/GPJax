from typing import Optional

from chex import dataclass
from multipledispatch import dispatch

from .kernels import Kernel
from .likelihoods import (Gaussian, Likelihood, NonConjugateLikelihoods,
                          NonConjugateLikelihoodType)
from .mean_functions import MeanFunction, Zero


@dataclass(repr=False)
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

    def __repr__(self):
        meanf_string = self.prior.mean_function.__repr__()
        kernel_string = self.prior.kernel.__repr__()
        likelihood_string = self.likelihood.__repr__()
        return f"Conjugate Posterior\n{'-'*80}\n- {meanf_string}\n- {kernel_string}\n- {likelihood_string}"


@dataclass
class NonConjugatePosterior:
    prior: Prior
    likelihood: NonConjugateLikelihoodType
    name: Optional[str] = "ConjugatePosterior"

    def __repr__(self):
        meanf_string = self.prior.mean_function.__repr__()
        kernel_string = self.prior.kernel.__repr__()
        likelihood_string = self.likelihood.__repr__()
        return f"Conjugate Posterior\n{'-'*80}\n- {meanf_string}\n- {kernel_string}\n- {likelihood_string}"
