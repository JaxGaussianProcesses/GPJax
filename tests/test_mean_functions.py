import typing as tp

import jax.numpy as jnp
import pytest
from numpy import isin

from gpjax.mean_functions import Constant, Zero
from gpjax.parameters import initialise


@pytest.mark.parametrize("meanf", [Zero, Constant])
@pytest.mark.parametrize("dim", [1, 2, 5])
def test_shape(meanf, dim):
    meanf = meanf(output_dim=dim)
    x = jnp.linspace(-1.0, 1.0, num=10).reshape(-1, 1)
    if dim > 1:
        x = jnp.hstack([x] * dim)
    params, _, _, _ = initialise(meanf)
    mu = meanf(x, params)
    assert mu.shape[0] == x.shape[0]
    assert mu.shape[1] == dim


@pytest.mark.parametrize("meanf", [Zero, Constant])
def test_initialisers(meanf):
    params, _, _, _ = initialise(meanf())
    assert isinstance(params, tp.Dict)
