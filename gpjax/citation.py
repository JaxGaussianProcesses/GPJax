from dataclasses import (
    dataclass,
    fields,
)
from functools import singledispatch

from beartype.typing import (
    Dict,
    Union,
)
from jaxlib.xla_extension import PjitFunction

from gpjax.decision_making.test_functions import (
    Forrester,
    LogarithmicGoldsteinPrice,
)
from gpjax.decision_making.utility_functions import ThompsonSampling
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

CitationType = Union[str, Dict[str, str]]


@dataclass(repr=False)
class AbstractCitation:
    citation_key: str = None
    authors: str = None
    title: str = None
    year: str = None

    def as_str(self) -> str:
        citation_str = f"@{self.citation_type}{{{self.citation_key},"
        for field in fields(self):
            fn = field.name
            if fn not in ["citation_type", "citation_key", "notes"]:
                citation_str += f"\n{fn} = {{{getattr(self, fn)}}},"
        return citation_str + "\n}"

    def __repr__(self) -> str:
        return repr(self.as_str())

    def __str__(self) -> str:
        return self.as_str()


class NullCitation(AbstractCitation):
    def __str__(self) -> str:
        return (
            "No citation available. If you think this is an error, please open a pull"
            " request."
        )


class JittedFnCitation(AbstractCitation):
    def __str__(self) -> str:
        return "Citation not available for jitted objects."


@dataclass
class PhDThesisCitation(AbstractCitation):
    school: str = None
    institution: str = None
    citation_type: str = "phdthesis"


@dataclass
class PaperCitation(AbstractCitation):
    booktitle: str = None
    citation_type: str = "inproceedings"


@dataclass
class BookCitation(AbstractCitation):
    publisher: str = None
    volume: str = None
    citation_type: str = "book"


####################
# Default citation
####################
@singledispatch
def cite(tree) -> AbstractCitation:
    return NullCitation()


####################
# Default citation
####################
@cite.register(PjitFunction)
def _(tree):
    return JittedFnCitation()


####################
# Kernel citations
####################
@cite.register(Matern12)
@cite.register(Matern32)
@cite.register(Matern52)
def _(tree) -> PhDThesisCitation:
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


@cite.register(ArcCosine)
def _(_) -> PaperCitation:
    return PaperCitation(
        citation_key="cho2009kernel",
        authors="Cho, Youngmin and Saul, Lawrence",
        title="Kernel Methods for Deep Learning",
        year="2009",
        booktitle="Advances in Neural Information Processing Systems",
    )


@cite.register(GraphKernel)
def _(tree) -> PaperCitation:
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


@cite.register(RFF)
def _(tree) -> PaperCitation:
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
@cite.register(ConjugateMLL)
@cite.register(NonConjugateMLL)
@cite.register(LogPosteriorDensity)
def _(tree) -> BookCitation:
    return BookCitation(
        citation_key="rasmussen2006gaussian",
        title="Gaussian Processes for Machine Learning",
        authors="Rasmussen, Carl Edward and Williams, Christopher K",
        year="2006",
        publisher="MIT press Cambridge, MA",
        volume="2",
    )


@cite.register(CollapsedELBO)
def _(tree) -> PaperCitation:
    return PaperCitation(
        citation_key="titsias2009variational",
        title="Variational learning of inducing variables in sparse Gaussian processes",
        authors="Titsias, Michalis",
        year="2009",
        booktitle="International Conference on Artificial Intelligence and Statistics",
    )


@cite.register(ELBO)
def _(tree) -> PaperCitation:
    return PaperCitation(
        citation_key="hensman2013gaussian",
        title="Gaussian Processes for Big Data",
        authors="Hensman, James and Fusi, Nicolo and Lawrence, Neil D",
        year="2013",
        booktitle="Uncertainty in Artificial Intelligence",
        citation_type="article",
    )


####################
# Decision making citations
####################
@cite.register(ThompsonSampling)
def _(tree) -> PaperCitation:
    return PaperCitation(
        citation_key="wilson2020efficiently",
        title="Efficiently sampling functions from Gaussian process posteriors",
        authors="Wilson, James and Borovitskiy, Viacheslav and Terenin, Alexander and Mostowsky, Peter and Deisenroth, Marc",
        year="2020",
        booktitle="International Conference on Machine Learning",
        citation_type="article",
    )


@cite.register(Forrester)
def _(tree) -> BookCitation:
    return BookCitation(
        citation_key="forrester2008engineering",
        authors="Forrester, Alexander and Sobester, Andras and Keane, Andy",
        title="Engineering design via surrogate modelling: a practical guide",
        year="2008",
        publisher="John Wiley & Sons",
    )


@cite.register(LogarithmicGoldsteinPrice)
def _(tree) -> PaperCitation:
    return PaperCitation(
        citation_key="picheny2013benchmark",
        authors="Picheny, Victor and Wagner, Tobias and Ginsbourger, David",
        title="A benchmark of kriging-based infill criteria for noisy optimization",
        year="2013",
        booktitle="Structural and multidisciplinary optimization",
        citation_type="article",
    )
