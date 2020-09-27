import matplotlib.pyplot as plt
from gp import Prior
from kernel import SquaredExponential
from mean_functions import Zero
from likelihoods import Gaussian
import numpy as np


if __name__=='__main__':
    x = np.linspace(-1, 1).reshape(-1, 1)
    k = SquaredExponential(lengthscale=[0.1])
    mfunc = Zero()
    gp = Prior(mfunc, k)
    mu, cov = gp.sample(x)
    samp =  np.random.multivariate_normal(mu.ravel(), cov, 10)
    # plt.plot(x, samp.T)
    # plt.show()

    lik = Gaussian()
    posterior = gp*lik
    print(posterior)