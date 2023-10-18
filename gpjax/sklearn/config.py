from dataclasses import dataclass
from functools import partial

from beartype.typing import (
    Optional,
    Protocol,
)
import jax.numpy as jnp
from jaxtyping import Num
from scipy.cluster.vq import kmeans2

from gpjax.typing import Array


class NumInducingHeuristic(Protocol):
    def __call__(self, n_data: int) -> int:
        ...


def default_inducing_heuristic(lower_limit: int, n_data: int) -> int:
    return min(n_data // 10, lower_limit)


class InducingPointSelector(Protocol):
    def __call__(
        self, full_data: Num[Array, "N D"], n_inducing: int
    ) -> Num[Array, "M D"]:
        ...


def kmeans_inducing_point_selector(
    full_data: Num[Array, "N D"], n_inducing: int
) -> Num[Array, "M D"]:
    return jnp.asarray(kmeans2(full_data, n_inducing, minit="points")[0])


@dataclass
class DefaultConfig:
    sparse_threshold: Optional[int] = 2000
    stochastic_threshold: Optional[int] = 20000
    min_num_inducing: Optional[int] = 100
    n_inducing_heuristic: Optional[NumInducingHeuristic] = None
    inducing_point_selector: Optional[InducingPointSelector] = None

    def __post_init__(self):
        if self.n_inducing_heuristic is None:
            self.n_inducing_heuristic = partial(
                default_inducing_heuristic, self.min_num_inducing
            )
        if self.inducing_point_selector is None:
            self.inducing_point_selector = kmeans_inducing_point_selector
