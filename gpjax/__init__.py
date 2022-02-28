from jax.config import config

# Enable Floa64 - this is crucial for more stable matrix inversions.
config.update("jax_enable_x64", True)
# Highlight any potentially unintended broadcasting rank promoting ops.
# config.update("jax_numpy_rank_promotion", "warn")

from .abstractions import fit, optax_fit
from .gps import Prior, construct_posterior, construct_VFE_posterior
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
from .types import Dataset

__version__ = "0.4.1"
