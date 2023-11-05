from abc import abstractmethod
from dataclasses import dataclass
import jax.numpy as jnp
from beartype.typing import Callable
from jaxtyping import Num, Float

from gpjax.typing import Array, ScalarFloat


@dataclass
class AbstractScore:
    name: str

    @abstractmethod
    def __call__(self, predictive_dist, y: Num[Array, "N 1"]) -> float:
        raise NotImplementedError("Please implement `__call__` in your subclass")


@dataclass
class SKLearnScore(AbstractScore):
    name: str
    fn: Callable[[Num[Array, "N 1"], Num[Array, "N 1"]], float]

    def __call__(self, predictive_dist, y: Num[Array, "N 1"]) -> ScalarFloat:
        mu = predictive_dist.mean()
        return jnp.array(self.fn(y, mu))


@dataclass
class LogPredictiveDensity(AbstractScore):
    name: str = "Log-posterior density"

    def __call__(self, predictive_dist, y: Num[Array, "N 1"]) -> ScalarFloat:
        return predictive_dist.log_prob(y).mean()
