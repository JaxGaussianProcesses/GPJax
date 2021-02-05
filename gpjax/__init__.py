from jax.config import config

config.update("jax_enable_x64", True)
from . import gps
from .kernel import RBF
from .likelihoods import Gaussian
from .mean_functions import ZeroMean
from .utilities import load, save

__version__ = "0.2.0"
