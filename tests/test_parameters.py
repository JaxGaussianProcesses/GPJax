import jax.numpy as jnp
from gpjax.parameters import Parameter
from gpjax.transforms import Softplus
import pytest


def hardcode_softplus(x: jnp.ndarray):
    return jnp.log(jnp.exp(x)-1.0)


@pytest.mark.parametrize("val", [0.5, 1.0])
def test_transform(val):
    v = jnp.array([val])
    x = Parameter(v, transform=Softplus)
    assert x.untransform == v
    print(f"xval: {x.value}")
    print(f"hcode: {hardcode_softplus(v)}")
    assert x.value == hardcode_softplus(v)
