import typing as tp

from flax.experimental import nnx
import jax.numpy as jnp
import jax.tree_util as jtu
from jax.typing import ArrayLike
import tensorflow_probability.substrates.jax.bijectors as tfb

T = tp.TypeVar("T", bound=tp.Union[ArrayLike, list[float]])


class Parameter(nnx.Variable[T]):
    """Parameter base class."""

    def __init__(self, value: T, **kwargs):
        if not isinstance(value, (ArrayLike, list)):
            raise TypeError(
                f"Expected parameter value to be an array-like type. Got {type(value)}."
            )
        value = jnp.asarray(value)
        super().__init__(value=value, **kwargs)


class PositiveReal(Parameter[T]):
    """Parameter that is strictly positive."""

    def __init__(self, value: T, **kwargs):
        value = jnp.asarray(value)

        if jnp.any(value < 0):
            raise ValueError(
                f"Expected parameter value to be strictly positive. Got {value}."
            )

        super().__init__(value=value, **kwargs)


class Real(Parameter[T]):
    """Parameter that can take any real value."""


class SigmoidBounded(Parameter[T]):
    """Parameter that is bounded between 0 and 1."""


class Static(nnx.Variable[T]):
    """Parameter that does not change during training."""

    def __init__(self, value: T, **kwargs):
        if not isinstance(value, (ArrayLike, list)):
            raise TypeError(
                f"Expected parameter value to be an array-like type. Got {type(value)}."
            )
        value = jnp.asarray(value)
        super().__init__(value=value, **kwargs)


class TransformedParameter(Parameter[T]):
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


DEFAULT_BIJECTION = {
    PositiveReal: tfb.Softplus(),
    Real: tfb.Identity(),
    SigmoidBounded: tfb.Sigmoid(low=0.0, high=1.0),
}


def transform(
    params: nnx.State,
    params_bijection: tp.Dict[tp.Type, tfb.Bijector],
    inverse: bool = False,
):
    def _inner(param: Parameter):
        bijector = params_bijection.get(type(param), tfb.Identity())

        if inverse:
            transformed_value = bijector.inverse(param.value)
        else:
            transformed_value = bijector.forward(param.value)

        return param.replace(value=transformed_value)

    params.update(
        jtu.tree_map(
            lambda x: _inner(x),
            params,
            is_leaf=lambda x: isinstance(x, Parameter),
        )
    )

    return params
