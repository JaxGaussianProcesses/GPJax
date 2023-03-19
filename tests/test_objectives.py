from gpjax.objectives import (
    AbstractObjective,
    LogPosteriorDensity,
    ConjugateMLL,
    NonConjugateMLL,
)
import gpjax as gpx
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


def build_data(num_datapoints: int, num_dims: int, key, binary: bool):
    x = jr.uniform(key=key, minval=-2.0, maxval=2.0, shape=(num_datapoints, num_dims))
    if binary:
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
    else:
        y = (
            jnp.sin(x[:, 1]).reshape(-1, 1)
            + jr.normal(key=key, shape=(num_datapoints, 1)) * 0.1
        )
    D = ju.Dataset(X=x, y=y)
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
    p = Prior(kernel=jk.RBF(active_dims=list(range(num_dims))))
    lik = Gaussian(num_datapoints=num_datapoints)
    post = p * lik
    params = post.init_params(key)

    mll = ConjugateMLL(posterior=post, negative=negative)
    assert isinstance(mll, AbstractObjective)

    # Test parameter initialisation
    obj_params = mll.init_params(key)
    assert isinstance(obj_params, ju.Parameters)

    if jit_compile:
        mll = jax.jit(mll)

    evaluation = mll(params, D)
    assert isinstance(evaluation, jax.Array)
    assert evaluation.shape == ()

    # Test parameter initialisation
    # mll_params = ConjugateMLL(model=post, negative=negative).init_params(key)
    # assert all(mll_params == params)


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
    p = Prior(kernel=jk.RBF(active_dims=list(range(num_dims))))
    lik = Bernoulli(num_datapoints=num_datapoints)
    post = p * lik
    params = post.init_params(key)

    with pytest.raises(ValueError):
        ConjugateMLL(posterior=post, negative=negative)

    mll = NonConjugateMLL(posterior=post, negative=negative)
    assert isinstance(mll, AbstractObjective)
    if jit_compile:
        mll = jax.jit(mll)

    evaluation = mll(params, D)
    assert isinstance(evaluation, jax.Array)
    assert evaluation.shape == ()

    mll2 = LogPosteriorDensity(posterior=post, negative=negative)

    # Test parameter initialisation
    obj_params = mll2.init_params(key)
    assert isinstance(obj_params, ju.Parameters)

    if jit_compile:
        mll2 = jax.jit(mll2)
    assert mll2(params, D) == evaluation


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

    p = Prior(kernel=jk.RBF(active_dims=list(range(num_dims))))
    lik = Gaussian(num_datapoints=num_datapoints)
    post = p * lik

    q = gpx.CollapsedVariationalGaussian(prior=p, likelihood=lik, inducing_inputs=z)

    negative_elbo = gpx.CollapsedELBO(
        posterior=post, variational_family=q, negative=negative
    )
    params = negative_elbo.init_params(key)

    assert isinstance(negative_elbo, AbstractObjective)
    assert isinstance(params, ju.Parameters)

    if jit_compile:
        negative_elbo = jax.jit(negative_elbo)

    evaluation = negative_elbo(params, D)
    assert isinstance(evaluation, jax.Array)
    assert evaluation.shape == ()

    bern_post = p * Bernoulli(num_datapoints=num_datapoints)
    with pytest.raises(TypeError):
        gpx.CollapsedELBO(posterior=bern_post, variational_family=q, negative=negative)


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

    p = Prior(kernel=jk.RBF(active_dims=list(range(num_dims))))
    if binary:
        lik = Bernoulli(num_datapoints=num_datapoints)
    else:
        lik = Gaussian(num_datapoints=num_datapoints)
    post = p * lik

    q = gpx.VariationalGaussian(prior=p, inducing_inputs=z)

    negative_elbo = gpx.ELBO(
        posterior=post,
        variational_family=q,
        num_datapoints=num_datapoints,
        negative=negative,
    )
    params = negative_elbo.init_params(key)

    assert isinstance(negative_elbo, AbstractObjective)
    assert isinstance(params, ju.Parameters)

    if jit_compile:
        negative_elbo = jax.jit(negative_elbo)

    evaluation = negative_elbo(params, D)
    assert isinstance(evaluation, jax.Array)
    assert evaluation.shape == ()
