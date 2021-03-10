import jax.numpy as jnp
import pytest

from gpjax.kernels.utils import scale, stretch


@pytest.mark.parametrize("scale_dim", [1, 2, 5])
def test_scale(scale_dim):
    base_param = jnp.ones(shape=(10, scale_dim))
    scaler = jnp.ones(shape=(scale_dim,)) * jnp.array(2.0)
    scaled_param = scale(base_param, scaler)
    assert base_param.shape == scaled_param.shape
    assert (scaled_param == jnp.array(0.5)).all()
    assert scaled_param.dtype == jnp.float64


@pytest.mark.parametrize("matrix_shape", [(10, 10), (10, 1), (1, 10), (10, 3), (3, 10)])
@pytest.mark.parametrize("stretch_factor", [0.1, 0.5, 1.0, 10.0])
def test_stretch(matrix_shape, stretch_factor):
    A = jnp.ones(shape=matrix_shape)
    stretched_A = stretch(A, stretch_factor)
    assert (stretched_A == stretch_factor).all()
    assert stretched_A.shape == A.shape == matrix_shape
    assert stretched_A.dtype == jnp.float64
