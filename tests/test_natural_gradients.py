import typing as tp

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import gpjax as gpx
from gpjax.abstractions import get_batch
from gpjax.natural_gradients import (
    _expectation_elbo,
    _rename_expectation_to_natural,
    _rename_natural_to_expectation,
    natural_gradients,
    natural_to_expectation,
)
from gpjax.parameters import recursive_items

key = jr.PRNGKey(123)


def get_data_and_gp(n_datapoints):
    x = jnp.linspace(-5.0, 5.0, n_datapoints).reshape(-1, 1)
    y = jnp.sin(x) + jr.normal(key=jr.PRNGKey(123), shape=x.shape) * 0.1
    D = gpx.Dataset(X=x, y=y)

    p = gpx.Prior(kernel=gpx.RBF())
    lik = gpx.Gaussian(num_datapoints=n_datapoints)
    post = p * lik
    return D, post, p


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

    _, posterior, prior = get_data_and_gp(10)

    z = jnp.linspace(-5.0, 5.0, 5 * dim).reshape(-1, dim)
    expectation_variational_family = (
        gpx.variational_families.ExpectationVariationalGaussian(
            prior=prior, inducing_inputs=z
        )
    )

    natural_variational_family = gpx.variational_families.NaturalVariationalGaussian(
        prior=prior, inducing_inputs=z
    )

    natural_svgp = gpx.StochasticVI(
        posterior=posterior, variational_family=natural_variational_family
    )
    expectation_svgp = gpx.StochasticVI(
        posterior=posterior, variational_family=expectation_variational_family
    )

    key = jr.PRNGKey(123)
    natural_params, *_ = gpx.initialise(natural_svgp, key).unpack()
    expectation_params, *_ = gpx.initialise(expectation_svgp, key).unpack()

    expectation_params_test = natural_to_expectation(natural_params, jitter=1e-6)

    assert (
        "expectation_vector"
        in expectation_params_test["variational_family"]["moments"].keys()
    )
    assert (
        "expectation_matrix"
        in expectation_params_test["variational_family"]["moments"].keys()
    )
    assert (
        expectation_params_test["variational_family"]["moments"][
            "expectation_vector"
        ].shape
        == expectation_params["variational_family"]["moments"][
            "expectation_vector"
        ].shape
    )
    assert (
        expectation_params_test["variational_family"]["moments"][
            "expectation_matrix"
        ].shape
        == expectation_params["variational_family"]["moments"][
            "expectation_matrix"
        ].shape
    )


from copy import deepcopy


def test_renaming():
    """
    Converts natural parameters to expectation parameters.
    Args:
        natural_moments: A dictionary of natural parameters.
        jitter (float): A small value to prevent numerical instability.
    Returns:
        tp.Dict: A dictionary of Gaussian moments under the expectation parameterisation.
    """

    _, posterior, prior = get_data_and_gp(10)

    z = jnp.linspace(-5.0, 5.0, 5).reshape(-1, 1)
    expectation_variational_family = (
        gpx.variational_families.ExpectationVariationalGaussian(
            prior=prior, inducing_inputs=z
        )
    )

    natural_variational_family = gpx.variational_families.NaturalVariationalGaussian(
        prior=prior, inducing_inputs=z
    )

    natural_svgp = gpx.StochasticVI(
        posterior=posterior, variational_family=natural_variational_family
    )
    expectation_svgp = gpx.StochasticVI(
        posterior=posterior, variational_family=expectation_variational_family
    )

    key = jr.PRNGKey(123)
    natural_params, *_ = gpx.initialise(natural_svgp, key).unpack()
    expectation_params, *_ = gpx.initialise(expectation_svgp, key).unpack()

    _nat = deepcopy(natural_params)
    _exp = deepcopy(expectation_params)

    rename_expectation_to_natural = _rename_expectation_to_natural(_exp)
    rename_natural_to_expectation = _rename_natural_to_expectation(_nat)

    # Check correct names are in the dictionaries:
    assert (
        "expectation_vector"
        in rename_natural_to_expectation["variational_family"]["moments"].keys()
    )
    assert (
        "expectation_matrix"
        in rename_natural_to_expectation["variational_family"]["moments"].keys()
    )
    assert (
        "natural_vector"
        not in rename_natural_to_expectation["variational_family"]["moments"].keys()
    )
    assert (
        "natural_matrix"
        not in rename_natural_to_expectation["variational_family"]["moments"].keys()
    )

    assert (
        "natural_vector"
        in rename_expectation_to_natural["variational_family"]["moments"].keys()
    )
    assert (
        "natural_matrix"
        in rename_expectation_to_natural["variational_family"]["moments"].keys()
    )
    assert (
        "expectation_vector"
        not in rename_expectation_to_natural["variational_family"]["moments"].keys()
    )
    assert (
        "expectation_matrix"
        not in rename_expectation_to_natural["variational_family"]["moments"].keys()
    )

    # Check the values are unchanged:
    for v1, v2 in zip(
        rename_natural_to_expectation["variational_family"]["moments"].values(),
        natural_params["variational_family"]["moments"].values(),
    ):
        assert jnp.all(v1 == v2)

    for v1, v2 in zip(
        rename_expectation_to_natural["variational_family"]["moments"].values(),
        expectation_params["variational_family"]["moments"].values(),
    ):
        assert jnp.all(v1 == v2)


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

    params, _, constrainer, unconstrainer = gpx.initialise(
        svgp, jr.PRNGKey(123)
    ).unpack()

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


def test_natural_gradients():
    """
    Tests the natural gradient and hyperparameter gradients.
    """
    D, p, prior = get_data_and_gp(10)

    z = jnp.linspace(-5.0, 5.0, 5).reshape(-1, 1)
    prior = gpx.Prior(kernel=gpx.RBF())
    q = gpx.NaturalVariationalGaussian(prior=prior, inducing_inputs=z)

    svgp = gpx.StochasticVI(posterior=p, variational_family=q)

    params, trainables, constrainers, unconstrainers = gpx.initialise(
        svgp, jr.PRNGKey(123)
    ).unpack()
    params = gpx.transform(params, unconstrainers)

    batch = get_batch(D, batch_size=10, key=jr.PRNGKey(42))

    nat_grads_fn, hyper_grads_fn = natural_gradients(svgp, D, constrainers, trainables)

    assert isinstance(nat_grads_fn, tp.Callable)
    assert isinstance(hyper_grads_fn, tp.Callable)

    val, nat_grads = nat_grads_fn(params, batch)
    val, hyper_grads = hyper_grads_fn(params, batch)

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
