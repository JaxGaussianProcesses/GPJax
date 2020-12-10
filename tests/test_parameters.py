import jax.numpy as jnp
from jax.nn import softplus
from gpjax.parameters import Parameter
import pytest


def hardcode_softplus(x: jnp.ndarray):
    return jnp.log(1. + jnp.exp(x))


@pytest.mark.parametrize("val", [1.0, 5.0, 10.])
def test_transform(val):
    x = Parameter(jnp.array([val]), transform=softplus)
    assert x.transformed == hardcode_softplus(val)