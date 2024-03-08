# Enable Float64 for more stable matrix inversions.
from jax import config

config.update("jax_enable_x64", True)


import jax.numpy as jnp
from jaxtyping import (
    Array,
    Float,
)
import pytest

from gpjax.mean_functions import (
    AbstractMeanFunction,
    Constant,
    Zero,
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


# TODO: rewrite this test after work on fit
# def test_zero_mean_remains_zero() -> None:
#     key = jr.PRNGKey(123)

#     x = jr.uniform(key=key, minval=0, maxval=1, shape=(20, 1))
#     y = jnp.full((20, 1), 50, dtype=jnp.float64)  # Dataset with non-zero mean
#     D = gpx.Dataset(X=x, y=y)

#     kernel = gpx.kernels.Constant(constant=jnp.array(0.0))
#     kernel = kernel.replace_trainable(
#         constant=False
#     )  # Prevent kernel from modelling non-zero mean
#     meanf = Zero()
#     prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
#     likelihood = gpx.likelihoods.Gaussian(
#         num_datapoints=D.n, obs_stddev=jnp.array(1e-3)
#     )
#     likelihood = likelihood.replace_trainable(obs_stddev=False)
#     posterior = prior * likelihood

#     negative_mll = gpx.objectives.ConjugateMLL(negative=True)
#     opt_posterior, _ = gpx.fit(
#         model=posterior,
#         objective=negative_mll,
#         train_data=D,
#         optim=ox.adam(learning_rate=0.5),
#         num_iters=1000,
#         safe=True,
#         key=key,
#     )

#     assert opt_posterior.prior.mean_function.constant == 0.0


def test_initialising_zero_mean_with_constant_raises_error():
    with pytest.raises(TypeError):
        Zero(constant=jnp.array([1.0]))
