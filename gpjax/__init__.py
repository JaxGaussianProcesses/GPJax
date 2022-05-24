from jax.config import config

# Enable Float64 - this is crucial for more stable matrix inversions.
config.update("jax_enable_x64", True)
# Highlight any potentially unintended broadcasting rank promoting ops.
# config.update("jax_numpy_rank_promotion", "warn")

from .abstractions import fit, fit_batches
from .gps import Prior, construct_posterior
from .kernels import (
    RBF,
    GraphKernel,
    Matern12,
    Matern32,
    Matern52,
    Polynomial,
    ProductKernel,
    SumKernel,
)
from .likelihoods import Bernoulli, Gaussian
from .mean_functions import Constant, Zero
from .parameters import copy_dict_structure, initialise, transform
from .sparse_gps import SVGP
from .types import Dataset
from .variational import VariationalGaussian, WhitenedVariationalGaussian

__version__ = "0.4.5"
