from objax import TrainVar
from objax.typing import JaxArray
from objax.variable import reduce_mean
from typing import Optional, Callable
from jax.nn import softplus
# from .utils import Transform, Identity


class Parameter(TrainVar):
    def __init__(self,
                 tensor: JaxArray,
                 reduce: Optional[Callable[[JaxArray],
                                           JaxArray]] = reduce_mean,
                 transform: Callable = softplus):
        """
        An extension of Objax's TrainVar object with the ability to apply a bijective transform to a parameter
        Args:
            tensor:
            reduce:
            transform:
        """
        super().__init__(tensor, reduce)
        self.fn = transform

    @property
    def transformed(self):
        return self.fn(self.value)
