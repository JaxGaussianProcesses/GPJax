from abc import abstractmethod
from dataclasses import dataclass
from typing import Any
from beartype.typing import Protocol, Callable
from gpjax.typing import Array
from jaxtyping import Num


@dataclass
class AbstractScore:
    name: str

    @abstractmethod
    def __call__(self, predictive_dist, y: Num[Array, "N Q"]) -> float:
        raise NotImplementedError("Please implement `__call__` in your subclass")


@dataclass
class SKLearnScore:
    name: str
    fn: Callable[[Num[Array, "N Q"], Num[Array, "N Q"]], float]

    def __call__(self, predictive_dist, y: Num[Array, "N Q"]) -> float:
        mu = predictive_dist.mean()
        return self.fn(y, mu)


@dataclass
class LogPredictiveDensity(AbstractScore):
    name: str = "Log-posterior density"

    def __call__(self, predictive_dist, y: Num[Array, "N Q"]) -> float:
        return predictive_dist.log_prob(y).sum()
