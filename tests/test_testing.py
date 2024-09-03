from hypothesis import (
    given,
    settings,
    strategies as st,
)
import jax.numpy as jnp

from gpjax.testing import (
    is_psd,
    sample_multivariate_gaussian_params,
    sample_univariate_gaussian_params,
    approx_equal,
)
from gpjax.typing import (
    MultivariateParams,
    UnivariateParams,
)


@given(params=sample_univariate_gaussian_params())
def test_univariate_sampling(params: UnivariateParams):
    mu, scale = params
    assert jnp.isreal(mu)
    assert scale > 0
    assert mu.shape == ()
    assert scale.shape == ()


@given(
    params=st.integers(min_value=1, max_value=20).flatmap(
        lambda n: sample_multivariate_gaussian_params(dim=n)
    )
)
def test_multivariate_validity(
    params: MultivariateParams,
):
    mu, cov = params
    assert jnp.all(jnp.isreal(mu))
    assert is_psd(cov)
    assert jnp.allclose(cov, cov.T)


@given(dim=st.integers(min_value=1, max_value=20), data=st.data())
def test_multivariate_shape(dim: int, data: st.DataObject):
    mu, cov = data.draw(sample_multivariate_gaussian_params(dim))
    assert mu.shape == (dim,)
    assert cov.shape == (dim, dim)


@given(dim=st.integers(min_value=1, max_value=20), data=st.data())
@settings(max_examples=3)
def test_is_psd(dim: int, data: st.DataObject):
    _, cov = data.draw(sample_multivariate_gaussian_params(dim))
    negative_identity = -1 + jnp.diagonal(cov) * -1.0
    cov += negative_identity
    assert not is_psd(cov)
