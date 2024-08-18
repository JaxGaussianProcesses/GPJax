import typing as tp

from flax import nnx
import jax.numpy as jnp
import jax.tree_util as jtu
from jax.typing import ArrayLike
import tensorflow_probability.substrates.jax.bijectors as tfb

T = tp.TypeVar("T", bound=tp.Union[ArrayLike, list[float]])
ParameterTag = str


def transform(
    params: nnx.State,
    params_bijection: tp.Dict[str, tfb.Bijector],
    inverse: bool = False,
) -> nnx.State:
    r"""Transforms parameters using a bijector.

    Example:
    ```pycon
        >>> from gpjax.parameters import PositiveReal, transform
        >>> import jax.numpy as jnp
        >>> import tensorflow_probability.substrates.jax.bijectors as tfb
        >>> from flax import nnx
        >>> params = nnx.State(
        >>>     {
        >>>         "a": PositiveReal(jnp.array([1.0])),
        >>>         "b": PositiveReal(jnp.array([2.0])),
        >>>     }
        >>> )
        >>> params_bijection = {'positive': tfb.Softplus()}
        >>> transformed_params = transform(params, params_bijection)
        >>> print(transformed_params["a"].value)
         [1.3132617]
    ```


    Args:
        params: A nnx.State object containing parameters to be transformed.
        params_bijection: A dictionary mapping parameter types to bijectors.
        inverse: Whether to apply the inverse transformation.

    Returns:
        State: A new nnx.State object containing the transformed parameters.
    """

    def _inner(param):
        bijector = params_bijection.get(param._tag, tfb.Identity())
        if inverse:
            transformed_value = bijector.inverse(param.value)
        else:
            transformed_value = bijector.forward(param.value)

        param = param.replace(transformed_value)
        return param

    gp_params, *other_params = params.split(Parameter, ...)

    transformed_gp_params: nnx.State = jtu.tree_map(
        lambda x: _inner(x),
        gp_params,
        is_leaf=lambda x: isinstance(x, nnx.VariableState),
    )
    return nnx.State.merge(transformed_gp_params, *other_params)


class Parameter(nnx.Variable[T]):
    """Parameter base class.

    All trainable parameters in GPJax should inherit from this class.

    """

    def __init__(self, value: T, tag: ParameterTag, **kwargs):
        _check_is_arraylike(value)

        super().__init__(value=jnp.asarray(value), **kwargs)
        self._tag = tag


class PositiveReal(Parameter[T]):
    """Parameter that is strictly positive."""

    def __init__(self, value: T, tag: ParameterTag = "positive", **kwargs):
        super().__init__(value=value, tag=tag, **kwargs)

        _check_is_positive(self.value)


class Real(Parameter[T]):
    """Parameter that can take any real value."""

    def __init__(self, value: T, tag: ParameterTag = "real", **kwargs):
        super().__init__(value, tag, **kwargs)


class SigmoidBounded(Parameter[T]):
    """Parameter that is bounded between 0 and 1."""

    def __init__(self, value: T, tag: ParameterTag = "sigmoid", **kwargs):
        super().__init__(value=value, tag=tag, **kwargs)

        _check_in_bounds(self.value, 0.0, 1.0)


class Static(nnx.Variable[T]):
    """Static parameter that is not trainable."""

    def __init__(self, value: T, tag: ParameterTag = "static", **kwargs):
        _check_is_arraylike(value)

        super().__init__(value=jnp.asarray(value), tag=tag, **kwargs)
        self._tag = tag


class LowerTriangular(Parameter[T]):
    """Parameter that is a lower triangular matrix."""

    def __init__(self, value: T, tag: ParameterTag = "lower_triangular", **kwargs):
        super().__init__(value=value, tag=tag, **kwargs)

        _check_is_square(self.value)
        _check_is_lower_triangular(self.value)


DEFAULT_BIJECTION = {
    "positive": tfb.Softplus(),
    "real": tfb.Identity(),
    "sigmoid": tfb.Sigmoid(low=0.0, high=1.0),
    "lower_triangular": tfb.FillTriangular(),
}


def _check_is_arraylike(value: T):
    if not isinstance(value, (ArrayLike, list)):
        raise TypeError(
            f"Expected parameter value to be an array-like type. Got {value}."
        )


def _check_is_positive(value: T):
    if jnp.any(value < 0):
        raise ValueError(
            f"Expected parameter value to be strictly positive. Got {value}."
        )


def _check_is_square(value: T):
    if value.shape[0] != value.shape[1]:
        raise ValueError(
            f"Expected parameter value to be a square matrix. Got {value}."
        )


def _check_is_lower_triangular(value: T):
    if not jnp.all(jnp.tril(value) == value):
        raise ValueError(
            f"Expected parameter value to be a lower triangular matrix. Got {value}."
        )


def _check_in_bounds(value: T, low: float, high: float):
    if jnp.any((value < low) | (value > high)):
        raise ValueError(
            f"Expected parameter value to be bounded between {low} and {high}. Got {value}."
        )
