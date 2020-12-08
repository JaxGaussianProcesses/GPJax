from objax import TrainVar
from objax.typing import JaxArray
from objax.variable import reduce_mean
from typing import Optional, Callable
from .utils import Transform, Identity
import jax.numpy as jnp


class Parameter(TrainVar):
    def __init__(self,
                 tensor: JaxArray,
                 reduce: Optional[Callable[[JaxArray],
                                           JaxArray]] = reduce_mean,
                 transform: Transform = Identity):
        """
        An extension of Objax's TrainVar object with the ability to apply a bijective transform to a parameter
        Args:
            tensor:
            reduce:
            transform:
        """
        unconstrained = transform.forward(tensor)
        super().__init__(unconstrained, reduce)
        self.transform = transform

    @property
    def constrained_value(self):
        return self.transform.reverse(self.value)
