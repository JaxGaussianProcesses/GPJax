from gpjax import Gaussian, Prior, RBF
from gpjax.gp import Posterior
import jax.numpy as jnp
import jax.random as jr
import pytest


def test_conjugate_posterior():
    p = Prior(RBF())
    lik = Gaussian()
    post = p*lik
    assert isinstance(post, Posterior)


@pytest.mark.parametrize('n', [1, 10])
def test_prior_samples(n):
    input_pts = jnp.linspace(-1., 1., 100).reshape(-1, 1)
    key = jr.PRNGKey(123)
    p = Prior(RBF())
    samples = p.sample(input_pts, key=key, n_samples=n)
    assert samples.shape == (n, 100)