from objax import nn, Module, TrainVar
from .kernel import Kernel
from .mean_functions import MeanFunction, ZeroMean
import jax.numpy as jnp
from .gp import Prior


class DTCPrior(Prior):
    def __init__(self,
                 inducing_points: jnp.ndarray,
                 kernel: Kernel,
                 mean_function: MeanFunction = ZeroMean(),
                 jitter: float = 1e-6):
        super().__init__(kernel, mean_function, jitter)
        self.inducing_points = inducing_points

    def computeQ(self, X):
        Kxu = self.kernel(X, self.inducing_points) # N \times m
        Kuu = self.kernel(self.inducing_points, self.inducing_points) # m \times m
        Q = jnp.matmul(jnp.matmul(Kxu, Kuu), Kxu.T)
        return Q


