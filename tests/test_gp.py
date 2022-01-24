from gpjax import likelihoods
from gpjax.parameters import initialise
import pytest

from gpjax.gps import (
    ConjugatePosterior,
    GP,
    NonConjugatePosterior,
    Prior,
)
from gpjax.kernels import RBF
from gpjax.likelihoods import Bernoulli, Gaussian, NonConjugateLikelihoods


@pytest.mark.parametrize("num_datapoints", [1, 10])
def test_conjugate_posterior(num_datapoints):
    p = Prior(kernel=RBF())
    lik = Gaussian(num_datapoints=num_datapoints)
    post = p * lik
    assert isinstance(post, ConjugatePosterior)
    assert isinstance(post, GP)
    assert isinstance(p, GP)


@pytest.mark.parametrize("num_datapoints", [1, 10])
@pytest.mark.parametrize("likelihood", NonConjugateLikelihoods)
def test_non_conjugate_poster(num_datapoints, likelihood):
    posterior = Prior(kernel=RBF()) * likelihood(num_datapoints=num_datapoints)
    assert isinstance(posterior, NonConjugatePosterior)
    assert isinstance(posterior, GP)


@pytest.mark.parametrize("num_datapoints", [1, 10])
@pytest.mark.parametrize("lik", [Bernoulli, Gaussian])
def test_param_construction(num_datapoints, lik):
    p = Prior(kernel=RBF()) * lik(num_datapoints=num_datapoints)
    params, _, _ = initialise(p)
    if isinstance(lik, Bernoulli):
        assert sorted(list(params.keys())) == [
            "kernel",
            "latent_fn",
            "likelihood",
            "mean_function",
        ]
    elif isinstance(lik, Gaussian):
        assert sorted(list(params.keys())) == [
            "kernel",
            "likelihood",
            "mean_function",
        ]
