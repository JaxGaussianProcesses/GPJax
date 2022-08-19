import typing as tp

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax as ox
import pytest
import tensorflow as tf
from jax import jit

import gpjax as gpx
from gpjax.natural_gradients import (
    _expectation_elbo,
    _stop_gradients_moments,
    _stop_gradients_nonmoments,
    natural_gradients,
    natural_to_expectation,
)
from gpjax.parameters import recursive_items

tf.random.set_seed(42)
key = jr.PRNGKey(123)


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_natural_to_expectation(dim):
    """
    Converts natural parameters to expectation parameters.
    Args:
        natural_moments: A dictionary of natural parameters.
        jitter (float): A small value to prevent numerical instability.
    Returns:
        tp.Dict: A dictionary of Gaussian moments under the expectation parameterisation.
    """

    natural_matrix = -0.5 * jnp.eye(dim)
    natural_vector = jnp.zeros((dim, 1))

    natural_moments = {
        "natural_matrix": natural_matrix,
        "natural_vector": natural_vector,
    }

    expectation_moments = natural_to_expectation(natural_moments, jitter=1e-6)

    assert "expectation_vector" in expectation_moments.keys()
    assert "expectation_matrix" in expectation_moments.keys()
    assert (
        expectation_moments["expectation_vector"].shape
        == natural_moments["natural_vector"].shape
    )
    assert (
        expectation_moments["expectation_matrix"].shape
        == natural_moments["natural_matrix"].shape
    )


def get_data_and_gp(n_datapoints):
    x = jnp.linspace(-5.0, 5.0, n_datapoints).reshape(-1, 1)
    y = jnp.sin(x) + jr.normal(key=jr.PRNGKey(123), shape=x.shape) * 0.1
    D = gpx.Dataset(X=x, y=y)

    p = gpx.Prior(kernel=gpx.RBF())
    lik = gpx.Gaussian(num_datapoints=n_datapoints)
    post = p * lik
    return D, post, p


@pytest.mark.parametrize("jit_fns", [True, False])
def test_expectation_elbo(jit_fns):
    """
    Tests the expectation ELBO.
    """
    D, posterior, prior = get_data_and_gp(10)

    z = jnp.linspace(-5.0, 5.0, 5).reshape(-1, 1)
    variational_family = gpx.variational_families.ExpectationVariationalGaussian(
        prior=prior, inducing_inputs=z
    )

    svgp = gpx.StochasticVI(posterior=posterior, variational_family=variational_family)

    params, _, constrainer, unconstrainer = gpx.initialise(svgp)

    expectation_elbo = _expectation_elbo(posterior, variational_family, D)

    if jit_fns:
        elbo_fn = jax.jit(expectation_elbo)
    else:
        elbo_fn = expectation_elbo

    assert isinstance(elbo_fn, tp.Callable)
    elbo_value = elbo_fn(params, D)
    assert isinstance(elbo_value, jnp.ndarray)

    # Test gradients
    grads = jax.grad(elbo_fn, argnums=0)(params, D)
    assert isinstance(grads, tp.Dict)
    assert len(grads) == len(params)


# def test_stop_gradients_nonmoments():
#     pass


# def test_stop_gradients_moments():
#     pass


def test_natural_gradients():
    """
    Tests the expectation ELBO.
    """
    D, p, prior = get_data_and_gp(10)

    z = jnp.linspace(-5.0, 5.0, 5).reshape(-1, 1)

    Dbatched = (
        D.cache().repeat().shuffle(D.n).batch(batch_size=128).prefetch(buffer_size=1)
    )

    likelihood = gpx.Gaussian(num_datapoints=D.n)
    prior = gpx.Prior(kernel=gpx.RBF())
    q = gpx.NaturalVariationalGaussian(prior=prior, inducing_inputs=z)

    svgp = gpx.StochasticVI(posterior=p, variational_family=q)

    params, trainables, constrainers, unconstrainers = gpx.initialise(svgp)
    params = gpx.transform(params, unconstrainers)

    batcher = Dbatched.get_batcher()
    batch = batcher()

    nat_grads_fn, hyper_grads_fn = natural_gradients(svgp, D, constrainers)

    assert isinstance(nat_grads_fn, tp.Callable)
    assert isinstance(hyper_grads_fn, tp.Callable)

    val, nat_grads = nat_grads_fn(params, trainables, batch)
    val, hyper_grads = hyper_grads_fn(params, trainables, batch)

    assert isinstance(val, jnp.ndarray)
    assert isinstance(nat_grads, tp.Dict)
    assert isinstance(hyper_grads, tp.Dict)

    # Need to check moments are zero in hyper_grads:
    assert jnp.array(
        [
            (v == 0.0).all()
            for v in hyper_grads["variational_family"]["moments"].values()
        ]
    ).all()

    # Check non-moments are zero in nat_grads:
    d = jax.tree_map(lambda x: (x == 0.0).all(), nat_grads)
    d["variational_family"]["moments"] = True

    assert jnp.array([v1 == True for k, v1, v2 in recursive_items(d, d)]).all()
