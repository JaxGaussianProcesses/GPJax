from abc import abstractmethod
from dataclasses import dataclass

from beartype.typing import Callable
from jaxtyping import Num

from gpjax.typing import Array


@dataclass
class AbstractScore:
    name: str

    @abstractmethod
    def __call__(self, predictive_dist, y: Num[Array, "N 1"]) -> float:
        raise NotImplementedError("Please implement `__call__` in your subclass")


@dataclass
class SKLearnScore:
    name: str
    fn: Callable[[Num[Array, "N 1"], Num[Array, "N 1"]], float]

    def __call__(self, predictive_dist, y: Num[Array, "N 1"]) -> float:
        mu = predictive_dist.mean()
        return self.fn(y, mu)


@dataclass
class LogPredictiveDensity(AbstractScore):
    name: str = "Log-posterior density"

    def __call__(self, predictive_dist, y: Num[Array, "N 1"]) -> float:
        return predictive_dist.log_prob(y).mean()
