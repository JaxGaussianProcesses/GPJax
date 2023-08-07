import jax
from jax.config import config
import jax.numpy as jnp
import jax.random as jr
import pytest

import gpjax as gpx
from gpjax import (
    Bernoulli,
    Gaussian,
    Prior,
)
from gpjax.dataset import Dataset
from gpjax.objectives import (
    ELBO,
    AbstractObjective,
    CollapsedELBO,
    ConjugateMLL,
    LogPosteriorDensity,
    NonConjugateMLL,
)

# Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)


def test_abstract_objective():
    with pytest.raises(TypeError):
        AbstractObjective()


def build_data(num_datapoints: int, num_dims: int, key, binary: bool):
    x = jr.uniform(key=key, minval=-2.0, maxval=2.0, shape=(num_datapoints, num_dims))
    if binary:
        y = (
            0.5
            * jnp.sign(
                jnp.cos(
                    3 * x[:, 0].reshape(-1, 1)
                    + jr.normal(key, shape=(num_datapoints, 1)) * 0.05
                )
            )
            + 0.5
        )
    else:
        y = (
            jnp.sin(x[:, 0]).reshape(-1, 1)
            + jr.normal(key=key, shape=(num_datapoints, 1)) * 0.1
        )
    D = Dataset(X=x, y=y)
    return D


@pytest.mark.parametrize("num_datapoints", [1, 2, 10])
@pytest.mark.parametrize("num_dims", [1, 2, 3])
@pytest.mark.parametrize("negative", [False, True])
@pytest.mark.parametrize("jit_compile", [False, True])
@pytest.mark.parametrize("key_val", [123, 42])
def test_conjugate_mll(
    num_datapoints: int, num_dims: int, negative: bool, jit_compile: bool, key_val: int
):
    key = jr.PRNGKey(key_val)
    D = build_data(num_datapoints, num_dims, key, binary=False)

    # Build model
    p = Prior(
        kernel=gpx.RBF(active_dims=list(range(num_dims))), mean_function=gpx.Constant()
    )
    likelihood = Gaussian(num_datapoints=num_datapoints)
    post = p * likelihood

    mll = ConjugateMLL(negative=negative)
    assert isinstance(mll, AbstractObjective)

    if jit_compile:
        mll = jax.jit(mll)

    evaluation = mll(post, D)
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
    D = build_data(num_datapoints, num_dims, key, binary=True)

    # Build model
    p = Prior(
        kernel=gpx.RBF(active_dims=list(range(num_dims))), mean_function=gpx.Constant()
    )
    likelihood = Bernoulli(num_datapoints=num_datapoints)
    post = p * likelihood

    mll = NonConjugateMLL(negative=negative)
    assert isinstance(mll, AbstractObjective)
    if jit_compile:
        mll = jax.jit(mll)

    evaluation = mll(post, D)
    assert isinstance(evaluation, jax.Array)
    assert evaluation.shape == ()

    mll2 = LogPosteriorDensity(negative=negative)

    if jit_compile:
        mll2 = jax.jit(mll2)
    assert mll2(post, D) == evaluation


@pytest.mark.parametrize("num_datapoints", [10, 20])
@pytest.mark.parametrize("num_dims", [1, 2, 3])
@pytest.mark.parametrize("negative", [False, True])
@pytest.mark.parametrize("jit_compile", [False, True])
@pytest.mark.parametrize("key_val", [123, 42])
def test_collapsed_elbo(
    num_datapoints: int, num_dims: int, negative: bool, jit_compile: bool, key_val: int
):
    key = jr.PRNGKey(key_val)
    D = build_data(num_datapoints, num_dims, key, binary=False)
    z = jr.uniform(
        key=key, minval=-2.0, maxval=2.0, shape=(num_datapoints // 2, num_dims)
    )

    p = Prior(
        kernel=gpx.RBF(active_dims=list(range(num_dims))), mean_function=gpx.Constant()
    )
    likelihood = Gaussian(num_datapoints=num_datapoints)
    q = gpx.CollapsedVariationalGaussian(posterior=p * likelihood, inducing_inputs=z)

    negative_elbo = CollapsedELBO(negative=negative)

    assert isinstance(negative_elbo, AbstractObjective)

    if jit_compile:
        negative_elbo = jax.jit(negative_elbo)

    evaluation = negative_elbo(q, D)
    assert isinstance(evaluation, jax.Array)
    assert evaluation.shape == ()

    # with pytest.raises(TypeError):


@pytest.mark.parametrize("num_datapoints", [1, 2, 10])
@pytest.mark.parametrize("num_dims", [1, 2, 3])
@pytest.mark.parametrize("negative", [False, True])
@pytest.mark.parametrize("jit_compile", [False, True])
@pytest.mark.parametrize("key_val", [123, 42])
@pytest.mark.parametrize("binary", [True, False])
def test_elbo(
    num_datapoints: int,
    num_dims: int,
    negative: bool,
    jit_compile: bool,
    key_val: int,
    binary: bool,
):
    key = jr.PRNGKey(key_val)
    D = build_data(num_datapoints, num_dims, key, binary=binary)
    z = jr.uniform(
        key=key, minval=-2.0, maxval=2.0, shape=(num_datapoints // 2, num_dims)
    )

    p = Prior(
        kernel=gpx.RBF(active_dims=list(range(num_dims))), mean_function=gpx.Constant()
    )
    if binary:
        likelihood = Bernoulli(num_datapoints=num_datapoints)
    else:
        likelihood = Gaussian(num_datapoints=num_datapoints)
    post = p * likelihood

    q = gpx.VariationalGaussian(posterior=post, inducing_inputs=z)

    negative_elbo = ELBO(
        negative=negative,
    )

    assert isinstance(negative_elbo, AbstractObjective)

    if jit_compile:
        negative_elbo = jax.jit(negative_elbo)

    evaluation = negative_elbo(q, D)
    assert isinstance(evaluation, jax.Array)
    assert evaluation.shape == ()
