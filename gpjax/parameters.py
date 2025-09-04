import typing as tp

from flax import nnx
import jax
from jax.experimental import checkify
import jax.numpy as jnp
from jax.typing import ArrayLike
import numpyro.distributions.transforms as npt

from gpjax.numpyro_extras import FillTriangularTransform

T = tp.TypeVar("T", bound=tp.Union[ArrayLike, list[float]])
ParameterTag = str


def transform(
    params: nnx.State,
    params_bijection: tp.Dict[str, npt.Transform],
    inverse: bool = False,
) -> nnx.State:
    r"""Transforms parameters using a bijector.

    This function uses JAX's functional tree_map with a robust is_leaf function
    that works across JAX versions 0.5.x through 0.7.x+.

    Example:
    ```pycon
        >>> from gpjax.parameters import PositiveReal, transform
        >>> import jax.numpy as jnp
        >>> import numpyro.distributions.transforms as npt
        >>> from flax import nnx
        >>> class TestModule(nnx.Module):
        ...     def __init__(self):
        ...         self.a = PositiveReal(jnp.array([1.0]))
        ...         self.b = PositiveReal(jnp.array([2.0]))
        >>> module = TestModule()
        >>> params = nnx.state(module)
        >>> params_bijection = {'positive': npt.SoftplusTransform()}
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
        # Handle the case where tree_map might pass JAX arrays instead of VariableState
        if not hasattr(param, "_tag"):
            # This might be a JAX array - skip transformation
            return param

        bijector = params_bijection.get(param._tag, npt.IdentityTransform())
        if inverse:
            transformed_value = bijector.inv(param.value)
        else:
            transformed_value = bijector(param.value)

        param = param.replace(transformed_value)
        return param

    # Use the new filter-based split API for NNX 0.11 compatibility
    gp_params, *other_params = params.split(Parameter, ...)

    # Use proper JAX tree_map with a more robust is_leaf function
    # This works across JAX versions by being very specific about what constitutes a leaf
    def is_parameter_leaf(x):
        # Treat Parameter-derived VariableState objects as leaves
        return (
            isinstance(x, nnx.VariableState)
            and hasattr(x, "_tag")
            and hasattr(x, "value")
        )

    # Try jax.tree.map first (JAX 0.7.0+), fallback to jax.tree_util.tree_map
    try:
        transformed_gp_params = jax.tree.map(
            _inner, gp_params, is_leaf=is_parameter_leaf
        )
    except AttributeError:
        # Fallback for older JAX versions
        import jax.tree_util as jtu

        transformed_gp_params = jtu.tree_map(
            _inner, gp_params, is_leaf=is_parameter_leaf
        )
    return nnx.State.merge(transformed_gp_params, *other_params)


class Parameter(nnx.Variable[T]):
    """Parameter base class.

    All trainable parameters in GPJax should inherit from this class.
    Compatible with NNX 0.11 module system and transforms.

    """

    def __init__(self, value: T, tag: ParameterTag, **kwargs):
        _check_is_arraylike(value)

        super().__init__(value=jnp.asarray(value), **kwargs)
        self._tag = tag


class NonNegativeReal(Parameter[T]):
    """Parameter that is non-negative."""

    def __init__(self, value: T, tag: ParameterTag = "non_negative", **kwargs):
        super().__init__(value=value, tag=tag, **kwargs)
        _safe_assert(_check_is_non_negative, self.value)


class PositiveReal(Parameter[T]):
    """Parameter that is strictly positive."""

    def __init__(self, value: T, tag: ParameterTag = "positive", **kwargs):
        super().__init__(value=value, tag=tag, **kwargs)
        _safe_assert(_check_is_positive, self.value)


class Real(Parameter[T]):
    """Parameter that can take any real value."""

    def __init__(self, value: T, tag: ParameterTag = "real", **kwargs):
        super().__init__(value, tag, **kwargs)


class SigmoidBounded(Parameter[T]):
    """Parameter that is bounded between 0 and 1."""

    def __init__(self, value: T, tag: ParameterTag = "sigmoid", **kwargs):
        super().__init__(value=value, tag=tag, **kwargs)

        # Only perform validation in non-JIT contexts
        if (
            not isinstance(value, jnp.ndarray)
            or getattr(value, "aval", None) is not None
        ):
            _safe_assert(
                _check_in_bounds,
                self.value,
                low=jnp.array(0.0),
                high=jnp.array(1.0),
            )


class Static(nnx.Variable[T]):
    """Static parameter that is not trainable.

    Compatible with NNX 0.11 module system and transforms.
    """

    def __init__(self, value: T, tag: ParameterTag = "static", **kwargs):
        _check_is_arraylike(value)

        super().__init__(value=jnp.asarray(value), **kwargs)
        self._tag = tag


class LowerTriangular(Parameter[T]):
    """Parameter that is a lower triangular matrix."""

    def __init__(self, value: T, tag: ParameterTag = "lower_triangular", **kwargs):
        super().__init__(value=value, tag=tag, **kwargs)

        # Only perform validation in non-JIT contexts
        if (
            not isinstance(value, jnp.ndarray)
            or getattr(value, "aval", None) is not None
        ):
            _safe_assert(_check_is_square, self.value)
            _safe_assert(_check_is_lower_triangular, self.value)


DEFAULT_BIJECTION = {
    "positive": npt.SoftplusTransform(),
    "non_negative": npt.SoftplusTransform(),
    "real": npt.IdentityTransform(),
    "sigmoid": npt.SigmoidTransform(),
    "lower_triangular": FillTriangularTransform(),
}


def _check_is_arraylike(value: T) -> None:
    """Check if a value is array-like.

    Args:
        value: The value to check.

    Raises:
        TypeError: If the value is not array-like.
    """
    if not isinstance(value, (ArrayLike, list)):
        raise TypeError(
            f"Expected parameter value to be an array-like type. Got {value}."
        )


@checkify.checkify
def _check_is_non_negative(value):
    checkify.check(
        jnp.all(value >= 0), "value needs to be non-negative, got {value}", value=value
    )


@checkify.checkify
def _check_is_positive(value):
    checkify.check(
        jnp.all(value > 0), "value needs to be positive, got {value}", value=value
    )


@checkify.checkify
def _check_is_square(value: T) -> None:
    """Check if a value is a square matrix.

    Args:
        value: The value to check.

    Raises:
        ValueError: If the value is not a square matrix.
    """
    checkify.check(
        value.shape[0] == value.shape[1],
        "value needs to be a square matrix, got {value}",
        value=value,
    )


@checkify.checkify
def _check_is_lower_triangular(value: T) -> None:
    """Check if a value is a lower triangular matrix.

    Args:
        value: The value to check.

    Raises:
        ValueError: If the value is not a lower triangular matrix.
    """
    checkify.check(
        jnp.all(jnp.tril(value) == value),
        "value needs to be a lower triangular matrix, got {value}",
        value=value,
    )


@checkify.checkify
def _check_in_bounds(value: T, low: T, high: T) -> None:
    """Check if a value is bounded between low and high.

    Args:
        value: The value to check.
        low: The lower bound.
        high: The upper bound.

    Raises:
        ValueError: If any element of value is outside the bounds.
    """
    checkify.check(
        jnp.all((value >= low) & (value <= high)),
        "value needs to be bounded between {low} and {high}, got {value}",
        value=value,
        low=low,
        high=high,
    )


def _safe_assert(fn: tp.Callable[[tp.Any], None], value: T, **kwargs) -> None:
    error, _ = fn(value, **kwargs)
    checkify.check_error(error)
