from beartype.typing import Union, Dict
from plum import dispatch

from gpjax.kernels import ArcCosine, GraphKernel, Matern12, Matern32, Matern52, RFF
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


def _citation_as_str(citation: Dict[str, str]) -> str:
    citation_str = f"@{citation['citation_type']}{{{citation['citation_key']},"
    for k, v in citation.items():
        if k not in ["citation_type", "citation_key"]:
            citation_str += f"\n{k} = {{{v}}},"
    return citation_str + "\n}"


@dispatch
def cite(tree) -> str:
    return (
        "No citation available. If you think this is an error, please open a pull"
        " request."
    )


####################
# Kernel citations
####################
# @dispatch
# def cite(tree: MaternKernels) -> str:
#     return """@phdthesis{matern1960SpatialV,
#     author      = {Bertil {M}atérn},
#     school      = {Stockholm University},
#     institution = {Stockholm University},
#     title       = {Spatial variation : Stochastic models and their application to some problems in forest surveys and other sampling investigations},
#     year        = {1960}
#     }"""


@dispatch
def cite(tree: MaternKernels, as_str: bool = False) -> CitationType:
    citation = {
        "author": "Bertil Matern",
        "school": "Stockholm University",
        "institution": "Stockholm University",
        "title": (
            "Spatial variation : Stochastic models and their application to some"
            " problems in forest surveys and other sampling investigations"
        ),
        "year": "1960",
        "citation_type": "phdthesis",
        "citation_key": "matern1960SpatialV",
        "notes": ""
    }
    if as_str:
        return _citation_as_str(citation)
    else:
        return citation


@dispatch
def cite(tree: ArcCosine) -> str:
    return """@inproceedings{cho2009kernel,
    author = {Cho, Youngmin and Saul, Lawrence},
    booktitle = {Advances in Neural Information Processing Systems},
    title = {Kernel Methods for Deep Learning},
    volume = {22},
    year = {2009}
    }"""


@dispatch
def cite(tree: GraphKernel) -> str:
    return """@inproceedings{borovitskiy2021matern,
    title     = {Matérn Gaussian processes on graphs},
    author    = {Borovitskiy, Viacheslav and Azangulov, Iskander and Terenin, Alexander and Mostowsky, Peter and Deisenroth, Marc and Durrande, Nicolas},
    booktitle = {International Conference on Artificial Intelligence and Statistics},
    year      = {2021}
    }"""


@dispatch
def cite(tree: RFF) -> str:
    return """@article{rahimi2007random,
    title={Random features for large-scale kernel machines},
    author={Rahimi, Ali and Recht, Benjamin},
    journal={Advances in neural information processing systems},
    volume={20},
    year={2007}
    }
    """


####################
# Objective citations
####################
@dispatch
def cite(tree: MLLs) -> str:
    return """@book{rasmussen2006gaussian,
    title     = {Gaussian processes for machine learning},
    author    = {Rasmussen, Carl Edward and Williams, Christopher K},
    volume    = {2},
    number    = {3},
    year      = {2006},
    publisher = {MIT press Cambridge, MA}
    }"""


@dispatch
def cite(tree: CollapsedELBO) -> str:
    return """@inproceedings{titsias2009variational,
    title        = {Variational learning of inducing variables in sparse Gaussian processes},
    author       = {Titsias, Michalis},
    booktitle    = {International Conference on Artificial Intelligence and Statistics},
    pages        = {567--574},
    year         = {2009},
    organization = {PMLR}
    }"""


@dispatch
def cite(tree: ELBO) -> str:
    return """@article{hensman2013gaussian,
    title   = {Gaussian processes for big data},
    author  = {Hensman, James and Fusi, Nicolo and Lawrence, Neil D},
    journal = {International Conference on Artificial Intelligence and Statistics},
    year    = {2013}
    }
    """
