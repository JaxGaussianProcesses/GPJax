import jax.numpy as jnp
from multipledispatch import dispatch
from ..gps import SpectralPosterior
from ..parameters.transforms import Transformation, SoftplusTransformation, untransform
from typing import Callable
from ..utils import I, concat_dictionaries
from jax.scipy.linalg import solve_triangular


@dispatch(SpectralPosterior)
def marginal_ll(
    gp: SpectralPosterior,
    transformation: Transformation = SoftplusTransformation,
    negative: bool = False,
) -> Callable:
    def mll(params: dict, x: jnp.DeviceArray, y: jnp.DeviceArray, static_params: dict = None):
        params = untransform(params, transformation)
        params = concat_dictionaries(params, static_params)
        m = gp.prior.kernel.num_basis
        phi = gp.prior.kernel._build_phi(x, params)
        A = (params['variance'] / m) * jnp.matmul(jnp.transpose(phi), phi) + params['obs_noise'] * I(2 * m)

        RT = jnp.linalg.cholesky(A)
        R = jnp.transpose(RT)

        RtiPhit = solve_triangular(RT, jnp.transpose(phi))
        # Rtiphity=RtiPhit*y_tr;
        Rtiphity = jnp.matmul(RtiPhit, y)

        out = 0.5/params['obs_noise']*(jnp.sum(jnp.square(y)) - params['variance']/m*jnp.sum(jnp.square(Rtiphity)))
        n = x.shape[0]

        out += jnp.sum(jnp.log(jnp.diag(R))) + (n/2.-m) * jnp.log(params['variance']) + n/2*jnp.log(2*jnp.pi)
        constant = jnp.array(-1.0) if negative else jnp.array(1.0)
        return constant*out.reshape()
    return mll