import pytest

from gpjax.gps import (
    ConjugatePosterior,
    NonConjugatePosterior,
    Posterior,
    Prior,
    SpectralPosterior,
)
from gpjax.kernels import RBF, to_spectral
from gpjax.likelihoods import Gaussian, NonConjugateLikelihoods


def test_conjugate_posterior():
    p = Prior(kernel=RBF())
    lik = Gaussian()
    post = p * lik
    assert isinstance(post, ConjugatePosterior)


@pytest.mark.parametrize("likelihood", NonConjugateLikelihoods)
def test_non_conjugate_poster(likelihood):
    posterior = Prior(kernel=RBF()) * likelihood()
    assert isinstance(posterior, NonConjugatePosterior)


def test_spectral():
    kernel = to_spectral(RBF(), 10)
    posterior = Prior(kernel=kernel) * Gaussian()
    assert isinstance(posterior, SpectralPosterior)
