from objax import TrainVar
from objax.typing import JaxArray
from objax.variable import reduce_mean
from typing import Optional, Callable
from jax.nn import softplus


class Parameter(TrainVar):
    """
    Base parameter class. This is a simple extension of the `TrainVar` class in Objax that enables parameter transforms
    and, in the future, prior distributions to be placed on the parameter in question.
    """
    def __init__(self,
                 tensor: JaxArray,
                 reduce: Optional[Callable[[JaxArray],
                                           JaxArray]] = reduce_mean,
                 transform: Callable = softplus):
        """
        Args:
            tensor: The initial value of the parameter
            reduce: A helper function for parallelisable calls.
            transform: The bijective transformation that should be applied to the parameter.
        """
        super().__init__(tensor, reduce)
        self.fn = transform

    @property
    def transformed(self) -> JaxArray:
        """
        Return the paramerter's transformed valued.
        """
        return self.fn(self.value)
