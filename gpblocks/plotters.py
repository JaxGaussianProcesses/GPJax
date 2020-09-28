from .samplers import sample
import matplotlib.pyplot as plt
from multipledispatch import dispatch
from .types import Posterior, Prior
import numpy as np
from .config import Colours


@dispatch(Prior, np.ndarray)
def plot(gp: Prior, x: np.ndarray, ax=None, n_samples=10):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    c = Colours('/home/thomas/Documents/GPBlocks/gpblocks/config_files/defaults.yaml')
    prior_samples = sample(gp, x, n_samples)
    ax.plot(x, prior_samples.T, color=c.primary, alpha=0.5)
    return ax


@dispatch(Posterior, np.ndarray)
def plot(gp: Posterior, x:np.ndarray, ax=None, n_samples=1):
    print("posterior plot")