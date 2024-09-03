from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
import jax.numpy as jnp
from jaxtyping import Float
from beartype import typing as tp
from jax._src.numpy.lax_numpy import _ScalarMeta as JAXDType
from gpjax.typing import (
    Array,
    MultivariateParams,
    UnivariateParams,
)

_JITTER = 1e-6


@st.composite
def stable_float(
    draw: st.DrawFn, min_value=-10, max_value=10, abs_threshold=1e-6
) -> float:
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
def sample_univariate_gaussian_params(draw: st.DrawFn) -> UnivariateParams:
    loc = jnp.array(draw(st.floats()), dtype=jnp.float64)
    scale = jnp.array(
        draw(st.floats(min_value=_JITTER, exclude_min=True)), dtype=jnp.float64
    )
    return loc, scale


@st.composite
def sample_multivariate_gaussian_params(
    draw: st.DrawFn, dim: int
) -> MultivariateParams:
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


def shape_strategy(
    min_length: int = 1, max_length=100, min_dims: int = 1, max_dims: int = 20
) -> st.SearchStrategy[tp.Tuple[int, int]]:
    return st.tuples(
        st.integers(min_length, max_length), st.integers(min_dims, max_dims)
    )


@st.composite
def sample_jax_array(
    draw: st.DrawFn,
    length: int = 1,
    dim: int = 1,
    min_value: int = -100,
    max_value: int = 100,
    dtype: JAXDType = jnp.float64,
    unique_vals: bool = True,
) -> Float[Array, "{length} {int}"]:
    elements_strategy = st.floats(
        min_value=min_value, max_value=max_value, allow_nan=False, allow_infinity=False
    )
    array_vals = jnp.array(
        draw(
            arrays(
                dtype=dtype,
                shape=(length, dim),
                elements=elements_strategy,
                unique=unique_vals,
            )
        ),
        dtype=dtype,
    )
    return array_vals


@st.composite
def sample_dynamic_jax_array_pair(
    draw: st.DrawFn,
    min_length: int = 1,
    max_length: int = 100,
    min_dims: int = 1,
    max_dims: int = 20,
    singleton_output: int = bool,
    min_value: int = -100,
    max_value: int = 100,
    dtype: JAXDType = jnp.float64,
    unique_vals: bool = True,
) -> tp.Tuple[Float[Array, "{length} {int}"], Float[Array, "{length} {int}"]]:
    length, x_dim = draw(shape_strategy(min_length, max_length, min_dims, max_dims))
    if singleton_output:
        y_dim = 1
    else:
        y_dim = x_dim
    X = draw(
        sample_jax_array(
            length=length,
            dim=x_dim,
            min_value=min_value,
            max_value=max_value,
            unique_vals=unique_vals,
            dtype=dtype,
        )
    )
    y = draw(
        sample_jax_array(
            length=length,
            dim=y_dim,
            min_value=min_value,
            max_value=max_value,
            unique_vals=unique_vals,
            dtype=dtype,
        )
    )
    return X, y
