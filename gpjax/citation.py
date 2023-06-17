"""Allocate bibtex citations to relevant objects in GPJax."""
from dataclasses import (
    dataclass,
    fields,
)

from beartype.typing import (
    Dict,
    Union,
)
from jaxlib.xla_extension import PjitFunction
from plum import dispatch

from gpjax.kernels import (
    RFF,
    ArcCosine,
    GraphKernel,
    Matern12,
    Matern32,
    Matern52,
)
from gpjax.objectives import (
    ELBO,
    CollapsedELBO,
    ConjugateMLL,
    LogPosteriorDensity,
    NonConjugateMLL,
)

MaternKernels = Union[Matern12, Matern32, Matern52]
MLLs = Union[ConjugateMLL, NonConjugateMLL, LogPosteriorDensity]
CitationType = Union[str, Dict[str, str]]


@dataclass(repr=False)
class AbstractCitation:
    r"""Base class for citations."""

    citation_key: str = None
    authors: str = None
    title: str = None
    year: str = None

    def as_str(self) -> str:
        r""" "Generate a bibtex citation string.

        Citations are stored as dataclasses. This method generates a bibtex citation
        string from the relevant fields.

        Returns:
            str: A bibtex string.
        """
        citation_str = f"@{self.citation_type}{{{self.citation_key},"
        for field in fields(self):
            fn = field.name
            if fn not in ["citation_type", "citation_key", "notes"]:
                citation_str += f"\n{fn} = {{{getattr(self, fn)}}},"
        return citation_str + "\n}"

    def __repr__(self) -> str:
        r"""Return a bibtex citation string."""
        return repr(self.as_str())

    def __str__(self) -> str:
        r"""Return a bibtex citation string."""
        return self.as_str()


class NullCitation(AbstractCitation):
    r"""Null citation for objects without citations.

    This class is used as the fallback citation for objects without citations.
    """

    def __str__(self) -> str:
        """Return a bibtex citation string."""
        return (
            "No citation available. If you think this is an error, please open a pull"
            " request."
        )


class JittedFnCitation(AbstractCitation):
    r"""Jitted functions citation.

    When a fucnction is jitted in JAX, it is not possible to recover the original
    function. This object makes it clear to users why generating a citation is not
    possible.
    """

    def __str__(self) -> str:
        """Return a bibtex citation string."""
        return "Citation not available for jitted objects."


@dataclass
class PhDThesisCitation(AbstractCitation):
    r"""Citation for a PhD thesis."""

    school: str = None
    institution: str = None
    citation_type: str = "phdthesis"


@dataclass
class PaperCitation(AbstractCitation):
    r"""Citation for conference papers and journal articles."""

    booktitle: str = None
    citation_type: str = "inproceedings"


@dataclass
class BookCitation(AbstractCitation):
    r"""Citation for books."""

    publisher: str = None
    volume: str = None
    citation_type: str = "book"


####################
# Default citation
####################
@dispatch
def cite(tree) -> NullCitation:
    r"""Fallback citation for objects without a reference."""
    return NullCitation()


####################
# Default citation
####################
@dispatch
def cite(tree: PjitFunction) -> JittedFnCitation:
    r"""Fallback citation for jitted objects."""
    return JittedFnCitation()


####################
# Kernel citations
####################
@dispatch
def cite(tree: MaternKernels) -> PhDThesisCitation:
    r"""Citation for Matern kernels."""
    citation = PhDThesisCitation(
        citation_key="matern1960SpatialV",
        authors="Bertil Matérn",
        title=(
            "Spatial variation : Stochastic models and their application to some"
            " problems in forest surveys and other sampling investigations"
        ),
        year="1960",
        school="Stockholm University",
        institution="Stockholm University",
    )
    return citation


@dispatch
def cite(tree: ArcCosine) -> PaperCitation:
    r"""Citation for ArcCosine kernels."""
    return PaperCitation(
        citation_key="cho2009kernel",
        authors="Cho, Youngmin and Saul, Lawrence",
        title="Kernel Methods for Deep Learning",
        year="2009",
        booktitle="Advances in Neural Information Processing Systems",
    )


@dispatch
def cite(tree: GraphKernel) -> PaperCitation:
    r"""Citation for Matérn graph kernels."""
    return PaperCitation(
        citation_key="borovitskiy2021matern",
        title="Matérn Gaussian Processes on Graphs",
        authors=(
            "Borovitskiy, Viacheslav and Azangulov, Iskander and Terenin, Alexander and"
            " Mostowsky, Peter and Deisenroth, Marc and Durrande, Nicolas"
        ),
        booktitle="International Conference on Artificial Intelligence and Statistics",
        year="2021",
    )


@dispatch
def cite(tree: RFF) -> PaperCitation:
    r"""Citation for Random Fourier Features kernel approximations."""
    return PaperCitation(
        citation_key="rahimi2007random",
        authors="Rahimi, Ali and Recht, Benjamin",
        title="Random features for large-scale kernel machines",
        year="2007",
        booktitle="Advances in neural information processing systems",
        citation_type="article",
    )


####################
# Objective citations
####################
@dispatch
def cite(tree: MLLs) -> BookCitation:
    r"""Citation for the GP marginal log-likelihood."""
    return BookCitation(
        citation_key="rasmussen2006gaussian",
        title="Gaussian Processes for Machine Learning",
        authors="Rasmussen, Carl Edward and Williams, Christopher K",
        year="2006",
        publisher="MIT press Cambridge, MA",
        volume="2",
    )


@dispatch
def cite(tree: CollapsedELBO) -> PaperCitation:
    r"""Citation for the collapsed evidence lower bound."""
    return PaperCitation(
        citation_key="titsias2009variational",
        title="Variational learning of inducing variables in sparse Gaussian processes",
        authors="Titsias, Michalis",
        year="2009",
        booktitle="International Conference on Artificial Intelligence and Statistics",
    )


@dispatch
def cite(tree: ELBO) -> PaperCitation:
    r"""Citation for the uncollapsed evidence lower bound."""
    return PaperCitation(
        citation_key="hensman2013gaussian",
        title="Gaussian Processes for Big Data",
        authors="Hensman, James and Fusi, Nicolo and Lawrence, Neil D",
        year="2013",
        booktitle="Uncertainty in Artificial Intelligence",
        citation_type="article",
    )
