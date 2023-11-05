from dataclasses import dataclass

import tensorflow_probability.substrates.jax as tfp

from gpjax.kernels import AbstractKernel
from gpjax.mean_functions import AbstractMeanFunction
from gpjax.sklearn.base import BaseEstimator

tfd = tfp.distributions


@dataclass
class GPJaxOptimizer(BaseEstimator):
    kernel: AbstractKernel
    mean_function: AbstractMeanFunction = None
    n_inducing: int = -1


GPJaxOptimiser = GPJaxOptimizer
