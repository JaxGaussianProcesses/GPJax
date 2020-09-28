from gpblocks.plotters import plot
from gpblocks.types import Prior
import numpy as np
from gpblocks.types import SquaredExponential
from gpblocks.mean_functions import Zero
import matplotlib.pyplot as plt


if __name__=='__main__':
    kern = SquaredExponential()
    mean_func = Zero()
    gp = Prior(mean_func, kern)
    x = np.linspace(-5, 5, 200).reshape(-1, 1)
    fig, ax = plt.subplots(figsize=(10, 6))
    plot(gp, x, ax=ax, n_samples=10)
    plt.show()