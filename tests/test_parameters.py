from flax import nnx
import jax.numpy as jnp
import pytest
from jax import jit

from gpjax.kernels import RBF

from gpjax.parameters import (
    DEFAULT_BIJECTION,
    LowerTriangular,
    Parameter,
    PositiveReal,
    Real,
    SigmoidBounded,
    Static,
    transform,
    _check_is_positive,
    _safe_assert,
)
from jax.experimental import checkify


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
    t_param1_expected = DEFAULT_BIJECTION[params["param1"]._tag].forward(value)
    assert jnp.allclose(t_params["param1"].value, t_param1_expected)
    assert jnp.allclose(t_params["param2"].value, 2.0)

    # Test inverse transformation
    it_params = transform(t_params, DEFAULT_BIJECTION, inverse=True)
    assert repr(it_params) == repr(params)


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
