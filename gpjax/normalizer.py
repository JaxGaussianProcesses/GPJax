

from dataclasses import dataclass
import warnings

from beartype.typing import Optional
from typing import List, Callable, Union
import jax.numpy as jnp
import jax
from jaxtyping import Num, Float
from simple_pytree import Pytree
from gpjax.typing import Array, ScalarFloat
import gpjax as gpx
from jax import vmap
import tensorflow_probability.substrates.jax.bijectors as tfb
from gpjax.base import Module, param_field,static_field


@dataclass
class Normalizer(Module):
    x: Float[Array, "N D"] = static_field(None)
    sinharcsinh_skewness: Float[Array, " D"] = param_field(jnp.array([0.0]))
    sinharcsinh_tailweight: Float[Array, " D"] = param_field(jnp.array([1.0]), bijector=tfb.Softplus(low=jnp.array(1e-5, dtype=jnp.float64)))
    standardizer_scale: Float[Array, " D"] = param_field(jnp.array([1.0]), bijector=tfb.Softplus(low=jnp.array(1e-5, dtype=jnp.float64)))
    standardizer_shift: Float[Array, " D"] = param_field(jnp.array([0.0]))
        
    """
    :param x: input to transform
    :param log: whether to log x first before applying flows of transformations
    :return: flows of transformations to match x to standard Gaussian
    """

    def get_bijector(self):
        return tfb.Chain([
            tfb.SinhArcsinh(
                skewness=self.sinharcsinh_skewness,
                tailweight=self.sinharcsinh_tailweight,
            ),
            tfb.Scale(
                scale=self.standardizer_scale,
            ),
            tfb.Shift(
                shift=self.standardizer_shift,
            ),
        ])


    def loss_fn(self, negative=False, log_prior: Optional[Callable] = None)->gpx.objectives.AbstractObjective:
        class KL(gpx.objectives.AbstractObjective):
            def step(
                self,
                posterior: Normalizer,
                train_data: gpx.Dataset,
            ) -> ScalarFloat:
                bij = posterior.get_bijector()
                return self.constant * (
                    jnp.mean((0.5 * jnp.square(bij(posterior.x))))
                    - jnp.mean(bij.forward_log_det_jacobian(posterior.x,event_ndims=0)))
        return KL(negative=negative)

