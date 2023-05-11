import jax.numpy as jnp
from jaxtyping import (
    Array,
    Float,
)
import pytest

from gpjax.mean_functions import (
    AbstractMeanFunction,
    Constant,
)


def test_abstract() -> None:
    # Check abstract mean function cannot be instantiated, as the `__call__` method is not defined.
    with pytest.raises(TypeError):
        AbstractMeanFunction()

    # Check a "dummy" mean function with defined abstract method, `__call__`, can be instantiated.
    class DummyMeanFunction(AbstractMeanFunction):
        def __call__(self, x: Float[Array, " D"]) -> Float[Array, "1"]:
            return jnp.array([1.0])

    mf = DummyMeanFunction()
    assert isinstance(mf, AbstractMeanFunction)
    assert (mf(jnp.array([1.0])) == jnp.array([1.0])).all()
    assert (mf(jnp.array([2.0, 3.0])) == jnp.array([1.0])).all()


@pytest.mark.parametrize(
    "constant", [jnp.array([0.0]), jnp.array([1.0]), jnp.array([3.0])]
)
def test_constant(constant: Float[Array, " Q"]) -> None:
    mf = Constant(constant=constant)

    assert isinstance(mf, AbstractMeanFunction)
    assert (mf(jnp.array([[1.0]])) == jnp.array([constant])).all()
    assert (mf(jnp.array([[2.0, 3.0]])) == jnp.array([constant])).all()
    assert (mf(jnp.array([[1.0], [2.0]])) == jnp.array([constant, constant])).all()
    assert (
        mf(jnp.array([[1.0, 2.0], [3.0, 4.0]])) == jnp.array([constant, constant])
    ).all()
