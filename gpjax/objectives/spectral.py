from typing import Callable

import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
from multipledispatch import dispatch

from ..gps import SpectralPosterior
from ..types import Dataset
from ..utils import I, concat_dictionaries


@dispatch(SpectralPosterior)
def marginal_ll(
    gp: SpectralPosterior,
    transform: Callable,
    negative: bool = False,
) -> Callable:
    def mll(params: dict, training: Dataset, static_params: dict = None):
        params = transform(params)
        x, y = training.X, training.y
        if static_params:
            params = concat_dictionaries(params, static_params)
        m = gp.prior.kernel.num_basis
        phi = gp.prior.kernel._build_phi(x, params)
        A = (params["variance"] / m) * jnp.matmul(jnp.transpose(phi), phi) + params[
            "obs_noise"
        ] * I(2 * m)

        RT = jnp.linalg.cholesky(A)
        R = jnp.transpose(RT)

        RtiPhit = solve_triangular(RT, jnp.transpose(phi))
        # Rtiphity=RtiPhit*y_tr;
        Rtiphity = jnp.matmul(RtiPhit, y)

        out = (
            0.5
            / params["obs_noise"]
            * (jnp.sum(jnp.square(y)) - params["variance"] / m * jnp.sum(jnp.square(Rtiphity)))
        )
        n = x.shape[0]

        out += (
            jnp.sum(jnp.log(jnp.diag(R)))
            + (n / 2.0 - m) * jnp.log(params["variance"])
            + n / 2 * jnp.log(2 * jnp.pi)
        )
        constant = jnp.array(-1.0) if negative else jnp.array(1.0)
        return constant * out.reshape()

    return mll
