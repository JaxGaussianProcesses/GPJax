import haiku as hk


class Prior:
    def __init__(self, mean_function, kernel):
        super().__init__()
        self.mean_func = mean_function
        self.kernel = kernel

    def __mul__(self, other):
        return Posterior(self, other)


class Posterior:
    def __init__(self, prior, likelihood):
        super().__init__()
        self.prior = prior
        self.likelihood = likelihood
