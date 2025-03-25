from flax import nnx
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
