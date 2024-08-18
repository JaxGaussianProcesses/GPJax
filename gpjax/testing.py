from beartype import typing as tp
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
import jax.numpy as jnp
from jaxtyping import Float

from gpjax.typing import (
    Array,
    ScalarFloat,
)

_JITTER = 1e-6


@st.composite
def sample_univariate_gaussian_params(draw) -> tp.Tuple[ScalarFloat, ScalarFloat]:
    loc = jnp.array(draw(st.floats()), dtype=jnp.float64)
    scale = jnp.array(
        draw(st.floats(min_value=_JITTER, exclude_min=True)), dtype=jnp.float64
    )
    return loc, scale


@st.composite
def sample_multivariate_gaussian_params(
    draw, dim: int
) -> tp.Tuple[Float[Array, " N"], Float[Array, "N N"]]:
    mean = draw(
        arrays(dtype=float, shape=dim, elements=st.floats(min_value=-10, max_value=10))
    )
    lower_vals = draw(
        arrays(
            dtype=float,
            shape=(dim, dim),
            elements=st.floats(min_value=-10, max_value=10),
        )
    )
    cov = jnp.dot(lower_vals, lower_vals.T)
    cov += jnp.eye(dim) * _JITTER
    return mean, cov


def is_psd(matrix: Float[Array, "N N"]) -> bool:
    # try:
    eig_vals = jnp.linalg.eigvals(matrix)
    psd_status = jnp.all(eig_vals > 0.0)
    # except jnp.linalg.Lin
    return psd_status
