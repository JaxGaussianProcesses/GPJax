import pytest

from gpjax.kernels import RBF
from gpjax.likelihoods import Gaussian, NonConjugateLikelihoods
from gpjax.gps import Prior, NonConjugatePosterior, ConjugatePosterior, Posterior


def test_conjugate_posterior():
    p = Prior(kernel = RBF())
    lik = Gaussian()
    post = p * lik
    assert isinstance(post, ConjugatePosterior)


@pytest.mark.parametrize('likelihood', NonConjugateLikelihoods)
def test_non_conjugate_poster(likelihood):
    posterior = Prior(kernel = RBF()) * likelihood()
    assert isinstance(posterior, NonConjugatePosterior)
