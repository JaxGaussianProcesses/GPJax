import jax
import pytest
from gpjax.mean_functions import AbstractMeanFunction, Constant
from jaxtyping import Array, Float


def test_abstract() -> None:
    # Check abstract mean function cannot be instantiated, as the `__call__` method is not defined.
    with pytest.raises(TypeError):
        AbstractMeanFunction()

    # Check a "dummy" mean funcion with defined abstract method, `__call__`, can be instantiated.
    class DummyMeanFunction(AbstractMeanFunction):
        def __call__(self, x: Float[Array, "D"]) -> Float[Array, "1"]:
            return jax.numpy.array([1.0])

    mf = DummyMeanFunction()
    assert isinstance(mf, AbstractMeanFunction)
    assert (mf(jax.numpy.array([1.0])) == jax.numpy.array([1.0])).all()
    assert (mf(jax.numpy.array([2.0, 3.0])) == jax.numpy.array([1.0])).all()


@pytest.mark.parametrize(
    "constant", [jax.numpy.array([0.0]), jax.numpy.array([1.0]), jax.numpy.array([3.0])]
)
def test_constant(constant: Float[Array, "Q"]) -> None:
    mf = Constant(constant=constant)

    assert isinstance(mf, AbstractMeanFunction)
    assert (mf(jax.numpy.array([1.0])) == constant).all()
    assert (mf(jax.numpy.array([2.0, 3.0])) == constant).all()
    assert (
        jax.vmap(mf)(jax.numpy.array([[1.0], [2.0]]))
        == jax.numpy.array([constant, constant])
    ).all()
    assert (
        jax.vmap(mf)(jax.numpy.array([[1.0, 2.0], [3.0, 4.0]]))
        == jax.numpy.array([constant, constant])
    ).all()
