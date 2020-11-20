from .kernels import Kernel
from .likelihoods import Likelihood
from .mean_functions import MeanFunction


class Prior:
    def __init__(self, mean_function: MeanFunction, kernel: Kernel):
        super().__init__()
        self.mean_func = mean_function
        self.kernel = kernel

    def __mul__(self, other: Likelihood):
        return Posterior(self, other)


class Posterior:
    def __init__(self, prior: Prior, likelihood: Likelihood):
        super().__init__()
        self.prior = prior
        self.likelihood = likelihood
