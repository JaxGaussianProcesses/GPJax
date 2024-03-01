import typing as tp

from flax.experimental import nnx
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.bijectors as tfb

from gpjax.typing import Array

T = tp.TypeVar("T")


class Parameter(nnx.Variable[T]):
    """Parameter base class."""

    def __init__(self, value: T, **kwargs):
        if not isinstance(value, (int, float, Array)):
            raise ValueError(
                f"Expected parameter value to be a scalar or array. Got {value}."
            )
        super().__init__(value=value, **kwargs)


class PositiveReal(Parameter):
    """Parameter that is strictly positive."""

    def __init__(self, value: T):
        if value < 0:
            raise ValueError(
                f"Expected parameter value to be strictly positive. Got {value}."
            )

        super().__init__(value=value)


class Real(Parameter):
    """Parameter that can take any real value."""


class SigmoidBounded(Parameter):
    """Parameter that is bounded between 0 and 1."""


class Static(nnx.Variable[T]):
    """Parameter that does not change during training."""


class TransformedParameter(Parameter):
    """Parameter that is transformed using a bijector."""

    bj = tfb.Bijector

    def create_value(self, value: T):
        return self.bj.inverse(value)

    def get_value(self) -> T:
        return self.bj.forward(self.value)

    def set_value(self, value: T):
        return self.replace(value=self.bj.forward(value))


# class SigmoidBounded(TransformedParameter[T]):
#     bj = tfb.Sigmoid(low=0.0, high=1.0)


# class SoftplusPositive(TransformedParameter[T]):
#     bj = tfb.Softplus()
