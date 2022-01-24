import imp

from jax.config import config

# Enable Floa64 - this is crucial for more stable matrix inversions.
config.update("jax_enable_x64", True)
# Highlight any potentially unintended broadcasting rank promoting ops.
# config.update("jax_numpy_rank_promotion", "warn")

from .gps import Prior, construct_posterior
from .kernels import RBF
from .likelihoods import Bernoulli, Gaussian
from .mean_functions import Constant, Zero
from .parameters import initialise, transform
from .types import Dataset

__version__ = "0.3.8"
