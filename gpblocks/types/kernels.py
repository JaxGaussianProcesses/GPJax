import tensorflow as tf
from ..utils import to_default_float
from gpflow.base import Module, Parameter


class Kernel(Module):
    def __init__(self, name=None):
        super().__init__(name=name)

    def func(self, D):
        raise NotImplementedError


class Stationary(Kernel):
    def __init__(self, lengthscale=[1.0], variance=1.0, name=None):
        super().__init__(name=name)
        self.lengthscale = Parameter([to_default_float(l) for l in lengthscale])
        self.variance = Parameter(to_default_float(variance))


class SquaredExponential(Stationary):
    def __init__(self, lengthscale=[1.0], variance=[1.0]):
        super().__init__(lengthscale=lengthscale,
                         variance=variance,
                         name="SquaredExponential")

    def func(self, D):
        return tf.square(self.variance) * tf.exp(
            -D / (2 * tf.square(self.lengthscale)))