import jax
from jax import config
import jax.numpy as jnp
import jax.random as jr
import pytest

import gpjax as gpx
from gpjax.dataset import Dataset
from gpjax.gps import Prior
from gpjax.likelihoods import Gaussian
from gpjax.objectives import (
    ELBO,
    AbstractObjective,
    CollapsedELBO,
    ConjugateLOOCV,
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
    key = jr.key(key_val)
    D = build_data(num_datapoints, num_dims, key, binary=False)

    # Build model
    p = gpx.gps.Prior(
        kernel=gpx.kernels.RBF(active_dims=list(range(num_dims))),
        mean_function=gpx.mean_functions.Constant(),
    )
    likelihood = gpx.likelihoods.Gaussian(num_datapoints=num_datapoints)
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
def test_conjugate_loocv(
    num_datapoints: int, num_dims: int, negative: bool, jit_compile: bool, key_val: int
):
    key = jr.key(key_val)
    D = build_data(num_datapoints, num_dims, key, binary=False)

    # Build model
    p = Prior(
        kernel=gpx.kernels.RBF(active_dims=list(range(num_dims))),
        mean_function=gpx.mean_functions.Constant(),
    )
    likelihood = Gaussian(num_datapoints=num_datapoints)
    post = p * likelihood

    loocv = ConjugateLOOCV(negative=negative)
    assert isinstance(loocv, AbstractObjective)

    if jit_compile:
        loocv = jax.jit(loocv)

    evaluation = loocv(post, D)
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
    key = jr.key(key_val)
    D = build_data(num_datapoints, num_dims, key, binary=True)

    # Build model
    p = gpx.gps.Prior(
        kernel=gpx.kernels.RBF(active_dims=list(range(num_dims))),
        mean_function=gpx.mean_functions.Constant(),
    )
    likelihood = gpx.likelihoods.Bernoulli(num_datapoints=num_datapoints)
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
    key = jr.key(key_val)
    D = build_data(num_datapoints, num_dims, key, binary=False)
    z = jr.uniform(
        key=key, minval=-2.0, maxval=2.0, shape=(num_datapoints // 2, num_dims)
    )

    p = gpx.gps.Prior(
        kernel=gpx.kernels.RBF(active_dims=list(range(num_dims))),
        mean_function=gpx.mean_functions.Constant(),
    )
    likelihood = gpx.likelihoods.Gaussian(num_datapoints=num_datapoints)
    q = gpx.variational_families.CollapsedVariationalGaussian(
        posterior=p * likelihood, inducing_inputs=z
    )

    negative_elbo = CollapsedELBO(negative=negative)

    assert isinstance(negative_elbo, AbstractObjective)

    if jit_compile:
        negative_elbo = jax.jit(negative_elbo)

    evaluation = negative_elbo(q, D)
    assert isinstance(evaluation, jax.Array)
    assert evaluation.shape == ()

    # Data on the full dataset should be the same as the marginal likelihood
    q = gpx.variational_families.CollapsedVariationalGaussian(
        posterior=p * likelihood, inducing_inputs=D.X
    )
    mll = ConjugateMLL(negative=negative)
    expected_value = mll(p * likelihood, D)
    actual_value = negative_elbo(q, D)
    assert jnp.abs(actual_value - expected_value) / expected_value < 1e-6


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
    key = jr.key(key_val)
    D = build_data(num_datapoints, num_dims, key, binary=binary)
    z = jr.uniform(
        key=key, minval=-2.0, maxval=2.0, shape=(num_datapoints // 2, num_dims)
    )

    p = gpx.gps.Prior(
        kernel=gpx.kernels.RBF(active_dims=list(range(num_dims))),
        mean_function=gpx.mean_functions.Constant(),
    )
    if binary:
        likelihood = gpx.likelihoods.Bernoulli(num_datapoints=num_datapoints)
    else:
        likelihood = gpx.likelihoods.Gaussian(num_datapoints=num_datapoints)
    post = p * likelihood

    q = gpx.variational_families.VariationalGaussian(posterior=post, inducing_inputs=z)

    negative_elbo = ELBO(
        negative=negative,
    )

    assert isinstance(negative_elbo, AbstractObjective)

    if jit_compile:
        negative_elbo = jax.jit(negative_elbo)

    evaluation = negative_elbo(q, D)
    assert isinstance(evaluation, jax.Array)
    assert evaluation.shape == ()
