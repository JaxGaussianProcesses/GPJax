from chex import dataclass


@dataclass
class AbstractObjective:
    def __call__(self, params):
        raise NotImplementedError


@dataclass
class MarginalLogLikelihood(AbstractObjective):
    def __call__(self, params):
        return super().__call__(params)
