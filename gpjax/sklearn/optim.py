from dataclasses import dataclass


from gpjax.kernels import AbstractKernel
from gpjax.mean_functions import AbstractMeanFunction
from gpjax.sklearn.base import BaseEstimator
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions


@dataclass
class GPJaxOptimizer(BaseEstimator):
    kernel: AbstractKernel
    mean_function: AbstractMeanFunction = None
    n_inducing: int = -1


GPJaxOptimiser = GPJaxOptimizer
