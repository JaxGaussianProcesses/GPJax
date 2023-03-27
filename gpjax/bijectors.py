from __future__ import annotations

__all__ = ["Bijector", "Identity", "Softplus"]

from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp
from simple_pytree import Pytree, static_field


@dataclass
class Bijector(Pytree):
    """
    Create a bijector.

    Args:
        forward(Callable): The forward transformation.
        inverse(Callable): The inverse transformation.

    Returns:
        Bijector: A bijector.
    """

    forward: Callable = static_field()
    inverse: Callable = static_field()


Identity = Bijector(forward=lambda x: x, inverse=lambda x: x)

Softplus = Bijector(
    forward=lambda x: jnp.log(1.0 + jnp.exp(x)),
    inverse=lambda x: jnp.log(jnp.exp(x) - 1.0),
)
