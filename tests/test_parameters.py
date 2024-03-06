from flax.experimental import nnx
import jax.numpy as jnp

from gpjax.parameters import (
    DEFAULT_BIJECTION,
    Parameter,
    PositiveReal,
    transform,
)


def test_transform():
    # Create mock parameters and bijectors
    params = nnx.State(
        {
            "param1": PositiveReal(1.0),
            "param2": Parameter(2.0),
        }
    )

    # Test forward transformation
    t_params = transform(params, DEFAULT_BIJECTION)
    t_param1_expected = DEFAULT_BIJECTION[PositiveReal].forward(1.0)
    assert jnp.allclose(t_params["param1"].value, t_param1_expected)
    assert jnp.allclose(t_params["param2"].value, 2.0)

    # Test inverse transformation
    t_params = transform(params, DEFAULT_BIJECTION, inverse=True)
    t_param1_expected = DEFAULT_BIJECTION[PositiveReal].inverse(
        t_params["param1"].value
    )
    assert jnp.allclose(t_params["param1"].value, 1.0)
    assert jnp.allclose(t_params["param2"].value, 2.0)
