from gpjax.objectives import (
    AbstractObjective,
    ConjugateMarginalLogLikelihood,
    NonConjugateMarginalLogLikelihood,
)
import pytest
import jax.random as jr
import jax.numpy as jnp
import jaxutils as ju
from gpjax import Prior, Gaussian, Bernoulli
import jaxkern as jk
import jax


def test_abstract_objective():
    with pytest.raises(TypeError):
        AbstractObjective()


@pytest.mark.parametrize("num_datapoints", [1, 2, 10])
@pytest.mark.parametrize("num_dims", [1, 2, 3])
@pytest.mark.parametrize("negative", [False, True])
@pytest.mark.parametrize("jit_compile", [False, True])
@pytest.mark.parametrize("key_val", [123, 42])
def test_conjugate_mll(
    num_datapoints: int, num_dims: int, negative: bool, jit_compile: bool, key_val: int
):
    key = jr.PRNGKey(key_val)
    x = jr.uniform(key=key, minval=-2.0, maxval=2.0, shape=(num_datapoints, num_dims))
    y = (
        jnp.sin(x[:, 1]).reshape(-1, 1)
        + jr.normal(key=key, shape=(num_datapoints, 1)) * 0.1
    )
    D = ju.Dataset(X=x, y=y)

    # Build model
    p = Prior(kernel=jk.RBF(active_dims=list(range(num_dims))))
    lik = Gaussian(num_datapoints=num_datapoints)
    post = p * lik
    params = post.init_params(key)

    mll = ConjugateMarginalLogLikelihood(model=post, negative=negative)
    assert isinstance(mll, AbstractObjective)
    if jit_compile:
        mll = jax.jit(mll)

    evaluation = mll(params, D)
    assert isinstance(evaluation, jax.Array)
    assert evaluation.shape == ()


@pytest.mark.parametrize("num_datapoints", [1, 2, 10])
@pytest.mark.parametrize("num_dims", [1, 2, 3])
@pytest.mark.parametrize("negative", [False, True])
@pytest.mark.parametrize("jit_compile", [False, True])
@pytest.mark.parametrize("key_val", [123, 42])
def test_non_conjugate_mll(
    num_datapoints: int, num_dims: int, negative: bool, jit_compile: bool, key_val: int
):
    key = jr.PRNGKey(key_val)
    x = jr.uniform(key=key, minval=-2.0, maxval=2.0, shape=(num_datapoints, num_dims))
    y = (
        0.5
        * jnp.sign(
            jnp.cos(
                3 * x[:, 1].reshape(-1, 1)
                + jr.normal(key, shape=(num_datapoints, 1)) * 0.05
            )
        )
        + 0.5
    )
    D = ju.Dataset(X=x, y=y)

    # Build model
    p = Prior(kernel=jk.RBF(active_dims=list(range(num_dims))))
    lik = Bernoulli(num_datapoints=num_datapoints)
    post = p * lik
    params = post.init_params(key)

    with pytest.raises(ValueError):
        ConjugateMarginalLogLikelihood(model=post, negative=negative)

    mll = NonConjugateMarginalLogLikelihood(model=post, negative=negative)
    assert isinstance(mll, AbstractObjective)
    if jit_compile:
        mll = jax.jit(mll)

    evaluation = mll(params, D)
    assert isinstance(evaluation, jax.Array)
    assert evaluation.shape == ()
