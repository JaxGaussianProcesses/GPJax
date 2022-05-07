import jax
import jax.numpy as jnp
import pytest

from gpjax.quadrature import gauss_hermite_quadrature


@pytest.mark.parametrize("jit", [True, False])
def test_quadrature(jit):
    def test():
        fun = lambda x: x ** 2
        mean = jnp.array([[2.0]])
        var = jnp.array([[1.0]])
        fn_val = gauss_hermite_quadrature(fun, mean, var)
        return fn_val.squeeze().round(1)

    if jit:
        test = jax.jit(test)
    assert test() == 5.0
