from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
import jax.numpy as jnp
from jaxtyping import Float

from gpjax.typing import (
    Array,
    MultivariateParams,
    UnivariateParams,
)

_JITTER = 1e-6


@st.composite
def stable_float(draw, min_value=-10, max_value=10, abs_threshold=1e-6) -> float:
    """
    Strategy to generate floats between min_value and max_value,
    excluding values with absolute value less than abs_threshold.

    Parameters:
    - min_value: minimum value for the float (default: -10)
    - max_value: maximum value for the float (default: 10)
    - abs_threshold: absolute values below this will be excluded (default: 1e-6)
    """
    value = draw(
        st.floats(
            min_value=min_value,
            max_value=max_value,
            allow_infinity=False,
            allow_nan=False,
        )
    )
    # If the value is too close to zero, move it outside the excluded range
    if abs(value) < abs_threshold:
        sign = 1 if value >= 0 else -1
        value = sign * (abs_threshold + (abs(value) % abs_threshold))
    return value


@st.composite
def sample_univariate_gaussian_params(draw) -> UnivariateParams:
    loc = jnp.array(draw(st.floats()), dtype=jnp.float64)
    scale = jnp.array(
        draw(st.floats(min_value=_JITTER, exclude_min=True)), dtype=jnp.float64
    )
    return loc, scale


@st.composite
def sample_multivariate_gaussian_params(draw, dim: int) -> MultivariateParams:
    mean = draw(
        arrays(
            dtype=float,
            shape=dim,
            elements=stable_float(min_value=-10, max_value=10, abs_threshold=1e-3),
        )
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


def approx_equal(
    res: Float[Array, "N N"], actual: Float[Array, "N N"], threshold: float = 1e-6
) -> bool:
    """Check if two arrays are approximately equal."""
    return jnp.linalg.norm(res - actual) < threshold
