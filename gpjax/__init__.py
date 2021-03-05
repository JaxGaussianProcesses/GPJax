from jax.config import config

config.update("jax_enable_x64", True)
from .gps import Prior
from .predict import mean, variance
from .sampling import random_variable, sample

__version__ = "0.3.2"
