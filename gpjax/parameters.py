import tensorflow_probability as tfp
from objax import TrainVar
from objax.typing import JaxArray
from objax.variable import reduce_mean
from typing import Optional, Callable
tfd = tfp.distributions


def identity(x: JaxArray):
    return x


class Parameter(TrainVar):
    def __init__(self,
                 tensor: JaxArray,
                 reduce: Optional[Callable[[JaxArray],
                                           JaxArray]] = reduce_mean,
                 transform: Callable = identity,
                 prior: tfp.distributions.Distribution = None):
        super().__init__(tensor, reduce)
        self.transform = transform
        self.prior = prior
