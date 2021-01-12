import jax.numpy as jnp
from jax.nn import softplus


class Transform:
    """
    Base class for all parameter transforms.
    """
    def __init__(self, name="Transformation"):
        self.name = name

    @staticmethod
    def forward(x: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    @staticmethod
    def backward(x: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError


class Softplus(Transform):
    """
    Softplus parameter transform
    """
    def __init__(self):
        super().__init__(name='Softplus')

    @staticmethod
    def forward(x: jnp.ndarray) -> jnp.ndarray:
        r"""
        Compute the transformation :math:`\log(\exp(x)-1)` that maps from a constrained into the entire real line.

        Args:
            x: The constrained array value

        Returns:
            An unconstrained form of the original array
        """
        return jnp.log(jnp.exp(x) - 1.)

    @staticmethod
    def backward(x: jnp.ndarray) -> jnp.ndarray:
        r"""
        Reverse the forward transformation and map back into a constrained space by :math:`\log(1+\exp(x))`.

        Args:
            x: The unconstrained array value

        Returns:
            A constrained form of the original array.
        """
        return softplus(x)


class Identity(Transform):
    """
    Identity parameter transform.
    """
    def __init__(self):
        super().__init__(name='Identity')

    @staticmethod
    def forward(x: jnp.ndarray) -> jnp.ndarray:
        r"""
         Compute the transformation :math:`y = g(x) = x`.

         Args:
             x: The input array value

         Returns:
             The identity transformed value
         """
        return x

    @staticmethod
    def backward(x: jnp.ndarray) -> jnp.ndarray:
        r"""
         Compute the transformation :math:`y = g(x) = x`.

         Args:
             x: The original transformed array value

         Returns:
             The untransformed array
         """
        return x
