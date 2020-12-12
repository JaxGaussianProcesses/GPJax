import jax.numpy as jnp
from jax.nn import softplus
from gpjax.parameters import Parameter
from gpjax.transforms import Softplus
import pytest


def hardcode_softplus(x: jnp.ndarray):
    return jnp.log(1. + jnp.exp(x))


@pytest.mark.parametrize("val", [1.0, 5.0])
def test_transform(val):
    x = Parameter(jnp.array([val]), transform=Softplus)
    assert x.transformed == val
    assert x.value == hardcode_softplus(val)
