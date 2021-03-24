from typing import Callable

import jax.numpy as jnp
import pytest
from numpy.testing import assert_array_equal

from gpjax import Prior
from gpjax.config import get_defaults
from gpjax.kernels import RBF
from gpjax.likelihoods import Bernoulli, Gaussian, Poisson
from gpjax.parameters import (
    build_all_transforms,
    build_constrain,
    build_unconstrain,
    initialise,
)


@pytest.mark.parametrize("transformation", [build_constrain, build_unconstrain])
@pytest.mark.parametrize("likelihood", [Gaussian, Poisson, Bernoulli])
def test_output(transformation, likelihood):
    posterior = Prior(kernel=RBF()) * likelihood()
    params = initialise(posterior, 10)
    config = get_defaults()
    transform_map = transformation(params.keys(), config)
    assert isinstance(transform_map, Callable)


@pytest.mark.parametrize("likelihood", [Gaussian, Poisson, Bernoulli])
def test_constrain(likelihood):
    posterior = Prior(kernel=RBF()) * likelihood()
    params = initialise(posterior, 10)
    config = get_defaults()
    transform_map = build_constrain(params.keys(), config)
    transformed_params = transform_map(params)
    assert transformed_params.keys() == params.keys()
    for u, v in zip(transformed_params.values(), params.values()):
        assert u.dtype == v.dtype


@pytest.mark.parametrize("likelihood", [Gaussian, Poisson, Bernoulli])
def test_unconstrain(likelihood):
    posterior = Prior(kernel=RBF()) * likelihood()
    params = initialise(posterior, 10)
    config = get_defaults()
    constrain_map = build_constrain(params.keys(), config)
    unconstrain_map = build_unconstrain(params.keys(), config)
    transformed_params = unconstrain_map(constrain_map(params))
    assert transformed_params.keys() == params.keys()
    for u, v in zip(transformed_params.values(), params.values()):
        assert_array_equal(u, v)
        assert u.dtype == v.dtype


@pytest.mark.parametrize("likelihood", [Gaussian, Poisson, Bernoulli])
def test_build_all_transforms(likelihood):
    posterior = Prior(kernel=RBF()) * likelihood()
    params = initialise(posterior, 10)
    config = get_defaults()
    t1, t2 = build_all_transforms(params.keys(), config)
    constrainer = build_constrain(params.keys(), config)
    constrained = t1(params)
    constrained2 = constrainer(params)
    assert constrained2.keys() == constrained2.keys()
    for u, v in zip(constrained.values(), constrained2.values()):
        assert_array_equal(u, v)
        assert u.dtype == v.dtype
    unconstrained = t2(params)
    unconstrainer = build_unconstrain(params.keys(), config)
    unconstrained2 = unconstrainer(params)
    for u, v in zip(unconstrained.values(), unconstrained2.values()):
        assert_array_equal(u, v)
        assert u.dtype == v.dtype
