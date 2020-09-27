from gpflow.base import Parameter, Module


class Likelihood(Module):
    def __init__(self, name=None):
        super().__init__(name=name)


class Gaussian(Likelihood):
    def __init__(self, observation_noise=1.0, name='Gaussian'):
        super().__init__(name=name)
        self.obs_noise = Parameter(observation_noise)