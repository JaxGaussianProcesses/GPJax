from typing import List

import jax.numpy as jnp
import pytest
from jaxtyping import Array, Float
from jaxkern.stationary.utils import euclidean_distance


@pytest.mark.parametrize(
    "a, b, distance_to_3dp",
    [
        ([1.0], [-4.0], 5.0),
        ([1.0, -2.0], [-4.0, 3.0], 7.071),
        ([1.0, 2.0, 3.0], [1.0, 1.0, 1.0], 2.236),
    ],
)
def test_euclidean_distance(
    a: List[float], b: List[float], distance_to_3dp: float
) -> None:

    # Convert lists to JAX arrays:
    a: Float[Array, "D"] = jnp.array(a)
    b: Float[Array, "D"] = jnp.array(b)

    # Test distance is correct to 3dp:
    assert jnp.round(euclidean_distance(a, b), 3) == distance_to_3dp
