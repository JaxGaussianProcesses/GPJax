from flax import nnx
from jax import jit
from jax.experimental import checkify
import jax.numpy as jnp
import pytest

from gpjax.parameters import (
    DEFAULT_BIJECTION,
    LowerTriangular,
    Parameter,
    PositiveReal,
    Real,
    SigmoidBounded,
    Static,
    _check_in_bounds,
    _check_is_lower_triangular,
    _check_is_positive,
    _check_is_square,
    _safe_assert,
    transform,
)


@pytest.mark.parametrize(
    "param, value",
    [
        (PositiveReal, 1.0),
        (Real, 2.0),
        (SigmoidBounded, 0.5),
    ],
)
def test_transform(param, value):
    # Create mock parameters and bijectors
    params = nnx.State(
        {
            "param1": param(value),
            "param2": Parameter(2.0, tag="real"),
        }
    )

    # Test forward transformation
    t_params = transform(params, DEFAULT_BIJECTION)
    t_param1_expected = DEFAULT_BIJECTION[params["param1"]._tag](value)
    assert jnp.allclose(t_params["param1"].value, t_param1_expected)
    assert jnp.allclose(t_params["param2"].value, 2.0)


@pytest.mark.parametrize(
    "param, tag",
    [
        (PositiveReal(1.0), "positive"),
        (Real(2.0), "real"),
        (SigmoidBounded(0.5), "sigmoid"),
        (Static(2.0), "static"),
        (LowerTriangular(jnp.eye(2)), "lower_triangular"),
    ],
)
def test_default_tags(param, tag):
    assert param._tag == tag


def test_check_is_positive():
    # Check singleton
    _safe_assert(_check_is_positive, jnp.array(3.0))
    # Check array
    _safe_assert(_check_is_positive, jnp.array([3.0, 4.0]))

    # Check negative singleton
    with pytest.raises(ValueError):
        _safe_assert(_check_is_positive, jnp.array(-3.0))

    # Check negative array
    with pytest.raises(ValueError):
        _safe_assert(_check_is_positive, jnp.array([-3.0, 4.0]))

    # Test that functions wrapping _check_is_positive are jittable
    def _dummy_fn(value):
        _safe_assert(_check_is_positive, value)

    jitted_fn = jit(checkify.checkify(_dummy_fn))
    jitted_fn(jnp.array(3.0))


def test_check_is_square():
    # Check square matrix
    _safe_assert(_check_is_square, jnp.full((2, 2), 1.0))
    # Check non-square matrix
    with pytest.raises(ValueError):
        _safe_assert(_check_is_square, jnp.full((2, 3), 1.0))


def test_check_is_lower_triangular():
    # Check lower triangular matrix
    _safe_assert(_check_is_lower_triangular, jnp.tril(jnp.eye(2)))
    # Check non-lower triangular matrix
    with pytest.raises(ValueError):
        _safe_assert(_check_is_lower_triangular, jnp.linspace(0.0, 1.0, 4))


def test_check_in_bounds():
    # Check in bounds
    _safe_assert(
        _check_in_bounds, jnp.array(0.5), low=jnp.array(0.0), high=jnp.array(1.0)
    )
    # Check out of bounds
    with pytest.raises(ValueError):
        _safe_assert(
            _check_in_bounds, jnp.array(1.5), low=jnp.array(0.0), high=jnp.array(1.0)
        )
