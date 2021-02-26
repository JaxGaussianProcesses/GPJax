from jax.config import config

config.update("jax_enable_x64", True)
from .gps import Prior
from .kernel import RBF, gram
from .likelihoods import Gaussian

__version__ = "0.3.0"
