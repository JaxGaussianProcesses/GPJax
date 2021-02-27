from gpjax.parameters.transforms import transform, untransform, SoftplusTransformation, IdentityTransformation
import pytest
import jax.numpy as jnp


@pytest.mark.parametrize('transformation', [IdentityTransformation, SoftplusTransformation])
@pytest.mark.parametrize('x', [1e-5, 0.01, 0.5, 1.0, 5.0])
def test_softplus(transformation, x):
    xarray = jnp.array(x)
    tr = transformation()
    transformed = tr.forward(tr.backward(x))
    assert pytest.approx(transformed, rel=1e-10) == xarray


@pytest.mark.parametrize('val', [0.01, 0.1, 1.0, 5.0])
@pytest.mark.parametrize('transformation', [IdentityTransformation, SoftplusTransformation])
def test_transform(transformation, val):
    params = {'lengthscale': jnp.array(val), 'variance': jnp.array(val), 'obs_noise': jnp.array(val)}
    transformed_params = transform(params, transformation())
    assert len(transformed_params.keys()) == len(params.keys())
    assert len(transformed_params.values()) == len(params.values())
    for v, p in zip(transformed_params.values(), params.values()):
        assert v == pytest.approx(transformation.forward(p), rel=1e-8)
    untransformed = untransform(transformed_params,transformation())
    for u, p in zip(untransformed.values(), params.values()):
        assert pytest.approx(u, rel=1e-8) == p
