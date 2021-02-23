from jax.config import config

config.update("jax_enable_x64", True)
from .kernel import RBF, gram
from .gps import Prior
from .likelihoods import Gaussian