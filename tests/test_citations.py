from jax.config import config

config.update("jax_enable_x64", True)

from jax import jit
import jax.numpy as jnp
import pytest

import gpjax as gpx
from gpjax.citation import (
    AbstractCitation,
    BookCitation,
    JittedFnCitation,
    NullCitation,
    PaperCitation,
    PhDThesisCitation,
    cite,
)
from gpjax.kernels import (
    RBF,
    RFF,
    ArcCosine,
    GraphKernel,
    Linear,
    Matern12,
    Matern32,
    Matern52,
)


def _check_no_fallback(citation: AbstractCitation):
    # Check the fallback has not been used
    assert repr(citation) != repr(
        "No citation available. If you think this is an error, please open a pull"
        " request."
    )
    assert (
        str(citation)
        != "No citation available. If you think this is an error, please open a pull"
        " request."
    )


@pytest.mark.parametrize("kernel", [Matern12(), Matern32(), Matern52()])
def test_matern_kernels(kernel):
    citation = cite(kernel)
    # Check type
    assert isinstance(citation, PhDThesisCitation)
    # Check some fields
    assert citation.authors == "Bertil Matérn"
    assert citation.year == "1960"
    assert citation.institution == "Stockholm University"

    _check_no_fallback(citation)


def test_arc_cosine():
    kernel = ArcCosine()
    citation = cite(kernel)

    assert isinstance(citation, PaperCitation)

    assert citation.citation_key == "cho2009kernel"
    assert citation.title == "Kernel Methods for Deep Learning"
    assert citation.authors == "Cho, Youngmin and Saul, Lawrence"
    _check_no_fallback(citation)


def test_graph_kernel():
    L = jnp.eye(3)
    kernel = GraphKernel(laplacian=L)
    citation = cite(kernel)

    assert isinstance(citation, PaperCitation)
    assert citation.citation_key == "borovitskiy2021matern"
    assert citation.title == "Matérn Gaussian Processes on Graphs"
    assert citation.year == "2021"
    _check_no_fallback(citation)


@pytest.mark.parametrize("kernel", [RBF(), Matern12(), Matern32(), Matern52()])
def test_rff(kernel):
    base_kernel = kernel
    rff = RFF(base_kernel=base_kernel)
    citation = cite(rff)

    assert isinstance(citation, PaperCitation)
    assert citation.citation_key == "rahimi2007random"
    assert citation.title == "Random features for large-scale kernel machines"
    assert citation.authors == "Rahimi, Ali and Recht, Benjamin"
    assert citation.year == "2007"
    _check_no_fallback(citation)


@pytest.mark.parametrize("kernel", [RBF(), Linear()])
def test_missing_citation(kernel):
    citation = cite(kernel)
    assert isinstance(citation, NullCitation)


@pytest.mark.parametrize(
    "mll", [gpx.ConjugateMLL(), gpx.NonConjugateMLL(), gpx.LogPosteriorDensity()]
)
def test_mlls(mll):
    citation = cite(mll)
    assert isinstance(citation, BookCitation)
    assert citation.citation_key == "rasmussen2006gaussian"
    assert citation.title == "Gaussian Processes for Machine Learning"
    assert citation.publisher == "MIT press Cambridge, MA"
    _check_no_fallback(citation)


def test_uncollapsed_elbo():
    elbo = gpx.ELBO()
    citation = cite(elbo)

    assert isinstance(citation, PaperCitation)
    assert citation.citation_key == "hensman2013gaussian"
    assert citation.title == "Gaussian Processes for Big Data"
    assert citation.authors == "Hensman, James and Fusi, Nicolo and Lawrence, Neil D"
    assert citation.year == "2013"
    assert citation.booktitle == "Uncertainty in Artificial Intelligence"
    _check_no_fallback(citation)


def test_collapsed_elbo():
    elbo = gpx.CollapsedELBO()
    citation = cite(elbo)

    assert isinstance(citation, PaperCitation)
    assert citation.citation_key == "titsias2009variational"
    assert (
        citation.title
        == "Variational learning of inducing variables in sparse Gaussian processes"
    )
    assert citation.authors == "Titsias, Michalis"
    assert citation.year == "2009"
    assert (
        citation.booktitle
        == "International Conference on Artificial Intelligence and Statistics"
    )
    _check_no_fallback(citation)


@pytest.mark.parametrize(
    "objective",
    [gpx.ELBO(), gpx.CollapsedELBO(), gpx.LogPosteriorDensity(), gpx.ConjugateMLL()],
)
def test_jitted_fallback(objective):
    citation = cite(jit(objective))
    assert isinstance(citation, JittedFnCitation)
    assert citation.__str__() == "Citation not available for jitted objects."
