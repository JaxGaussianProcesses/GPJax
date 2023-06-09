from plum import dispatch
from .kernels import Matern12, Matern32, Matern52, ArcCosine, GraphKernel
from .objectives import (
    ConjugateMLL,
    NonConjugateMLL,
    CollapsedELBO,
    ELBO,
    LogPosteriorDensity,
)
from typing import Union


MaternKernels = Union[Matern12, Matern32, Matern52]
MLLs = Union[ConjugateMLL, NonConjugateMLL, LogPosteriorDensity]


@dispatch
def cite(tree) -> str:
    return (
        "No citation available. If you think this is an error, please open a pull"
        " request."
    )


####################
# Kernel citations
####################
@dispatch
def cite(tree: MaternKernels) -> str:
    return """@phdthesis{matern1960SpatialV,
    author      = {Bertil {M}atérn},
    school      = {Stockholm University},
    institution = {Stockholm University},
    title       = {Spatial variation : Stochastic models and their application to some problems in forest surveys and other sampling investigations},
    year        = {1960}
    }"""


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
    title     = {Matérn {G}aussian processes on graphs},
    author    = {Borovitskiy, Viacheslav and Azangulov, Iskander and Terenin, Alexander and Mostowsky, Peter and Deisenroth, Marc and Durrande, Nicolas},
    booktitle = {International Conference on Artificial Intelligence and Statistics},
    year      = {2021}
    }"""


####################
# Objective citations
####################
@dispatch
def cite(tree: MLLs) -> str:
    return """@book{rasmussen2006gaussian,
    title     = {{G}aussian processes for machine learning},
    author    = {Rasmussen, Carl Edward and Williams, Christopher K},
    volume    = {2},
    number    = {3},
    year      = {2006},
    publisher = {MIT press Cambridge, MA}
    }"""


@dispatch
def cite(tree: CollapsedELBO) -> str:
    return """@inproceedings{titsias2009variational,
    title        = {Variational learning of inducing variables in sparse {G}aussian processes},
    author       = {Titsias, Michalis},
    booktitle    = {International Conference on Artificial Intelligence and Statistics},
    pages        = {567--574},
    year         = {2009},
    organization = {PMLR}
    }"""


@dispatch
def cite(tree: ELBO) -> str:
    return """@article{hensman2013gaussian,
    title   = {{G}aussian processes for big data},
    author  = {Hensman, James and Fusi, Nicolo and Lawrence, Neil D},
    journal = {International Conference on Artificial Intelligence and Statistics},
    year    = {2013}
    }
    """
