import jax.numpy as jnp
import pytest

from mytree.bijectors import Bijector, Identity, Softplus


def test_bijector():
    bij = Bijector(forward=lambda x: jnp.exp(x), inverse=lambda x: jnp.log(x))
    assert bij.forward(1.0) == pytest.approx(jnp.exp(1.0))
    assert bij.inverse(jnp.exp(1.0)) == pytest.approx(1.0)


def test_identity():
    bij = Identity
    assert bij.forward(1.0) == pytest.approx(1.0)
    assert bij.inverse(1.0) == pytest.approx(1.0)


def test_softplus():
    bij = Softplus
    assert bij.forward(1.0) == pytest.approx(jnp.log(1.0 + jnp.exp(1.0)))
    assert bij.inverse(jnp.log(1.0 + jnp.exp(1.0))) == pytest.approx(1.0)
