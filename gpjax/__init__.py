from jax.config import config
config.update("jax_enable_x64", True)
from gpjax.gps.priors import Prior
from .kernel import RBF
from .likelihoods import Gaussian
from .mean_functions import ZeroMean