# Copyright 2022 The GPJax Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


from beartype.typing import (
    Any,
    Generic,
    Optional,
    Tuple,
    TypeVar,
    Union,
)
import cola
from cola.ops import (
    Dense,
    Identity,
    LinearOperator,
)
from jax import vmap
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import (
    Bool,
    Float,
)
import tensorflow_probability.substrates.jax as tfp

from gpjax.lower_cholesky import lower_cholesky
from gpjax.typing import (
    Array,
    KeyArray,
    ScalarFloat,
)

tfd = tfp.distributions

from cola.linalg.decompositions.decompositions import Cholesky


def _check_loc_scale(loc: Optional[Any], scale: Optional[Any]) -> None:
    r"""Checks that the inputs are correct."""
    if loc is None and scale is None:
        raise ValueError("At least one of `loc` or `scale` must be specified.")

    if loc is not None and loc.ndim < 1:
        raise ValueError("The parameter `loc` must have at least one dimension.")

    if scale is not None and len(scale.shape) < 2:  # scale.ndim < 2:
        raise ValueError(
            "The `scale` must have at least two dimensions, but "
            f"`scale.shape = {scale.shape}`."
        )

    if scale is not None and not isinstance(scale, LinearOperator):
        raise ValueError(
            f"The `scale` must be a CoLA LinearOperator but got {type(scale)}"
        )

    if scale is not None and (scale.shape[-1] != scale.shape[-2]):
        raise ValueError(
            f"The `scale` must be a square matrix, but `scale.shape = {scale.shape}`."
        )

    if loc is not None:
        num_dims = loc.shape[-1]
        if scale is not None and (scale.shape[-1] != num_dims):
            raise ValueError(
                f"Shapes are not compatible: `loc.shape = {loc.shape}` and "
                f"`scale.shape = {scale.shape}`."
            )


class GaussianDistribution(tfd.Distribution):
    r"""Multivariate Gaussian distribution with a linear operator scale matrix.

    Args:
        loc (Optional[Float[Array, " N"]]): The mean of the distribution. Defaults to None.
        scale (Optional[LinearOperator]): The scale matrix of the distribution. Defaults to None.

    Returns
    -------
        GaussianDistribution: A multivariate Gaussian distribution with a linear operator scale matrix.
    """

    # TODO: Consider `distrax.transformed.Transformed` object. Can we create a LinearOperator to `distrax.bijector` representation
    # and modify `distrax.MultivariateNormalFromBijector`?

    # TODO: Consider natural and expectation parameterisations in future work.

    def __init__(
        self,
        loc: Optional[Float[Array, " N"]] = None,
        scale: Optional[LinearOperator] = None,
    ) -> None:
        r"""Initialises the distribution."""
        _check_loc_scale(loc, scale)

        # Find dimensionality of the distribution.
        if loc is not None:
            num_dims = loc.shape[-1]

        elif scale is not None:
            num_dims = scale.shape[-1]

        # Set the location to zero vector if unspecified.
        if loc is None:
            loc = jnp.zeros((num_dims,))

        # If not specified, set the scale to the identity matrix.
        if scale is None:
            scale = Identity(shape=(num_dims, num_dims), dtype=loc.dtype)

        self.loc = loc
        self.scale = cola.PSD(scale)

    def mean(self) -> Float[Array, " N"]:
        r"""Calculates the mean."""
        return self.loc

    def median(self) -> Float[Array, " N"]:
        r"""Calculates the median."""
        return self.loc

    def mode(self) -> Float[Array, " N"]:
        r"""Calculates the mode."""
        return self.loc

    def covariance(self) -> Float[Array, "N N"]:
        r"""Calculates the covariance matrix."""
        return self.scale.to_dense()

    def variance(self) -> Float[Array, " N"]:
        r"""Calculates the variance."""
        return cola.diag(self.scale)

    def stddev(self) -> Float[Array, " N"]:
        r"""Calculates the standard deviation."""
        return jnp.sqrt(cola.diag(self.scale))

    @property
    def event_shape(self) -> Tuple:
        r"""Returns the event shape."""
        return self.loc.shape[-1:]

    def entropy(self) -> ScalarFloat:
        r"""Calculates the entropy of the distribution."""
        return 0.5 * (
            self.event_shape[0] * (1.0 + jnp.log(2.0 * jnp.pi))
            + cola.logdet(self.scale, Cholesky(), Cholesky())
        )

    def log_prob(
        self, y: Float[Array, " N"], mask: Optional[Bool[Array, " N"]] = None
    ) -> ScalarFloat:
        r"""Calculates the log pdf of the multivariate Gaussian.

        Args:
            y (Optional[Float[Array, " N"]]): the value of which to calculate the log probability.
            mask: (Optional[Bool[Array, " N"]]): the mask for missing values in y.

        Returns
        -------
            ScalarFloat: The log probability of the value.
        """
        mu = self.loc
        sigma = self.scale
        n = mu.shape[-1]

        if mask is not None:
            y = jnp.where(mask, 0.0, y)
            mu = jnp.where(mask, 0.0, mu)
            sigma_masked = jnp.where(mask[None] + mask[:, None], 0.0, sigma.to_dense())
            sigma = cola.PSD(
                Dense(jnp.where(jnp.diag(mask), 1 / (2 * jnp.pi), sigma_masked))
            )

        # diff, y - µ
        diff = y - mu

        # compute the pdf, -1/2[ n log(2π) + log|Σ| + (y - µ)ᵀΣ⁻¹(y - µ) ]
        return -0.5 * (
            n * jnp.log(2.0 * jnp.pi)
            + cola.logdet(sigma, Cholesky(), Cholesky())
            + diff.T @ cola.solve(sigma, diff, Cholesky())
        )

    def _sample_n(self, key: KeyArray, n: int) -> Float[Array, "n N"]:
        r"""Samples from the distribution.

        Args:
            key (KeyArray): The key to use for sampling.

        Returns
        -------
            Float[Array, "n N"]: The samples.
        """
        # Obtain covariance root.
        sqrt = lower_cholesky(self.scale)

        # Gather n samples from standard normal distribution Z = [z₁, ..., zₙ]ᵀ.
        Z = jr.normal(key, shape=(n, *self.event_shape))

        # xᵢ ~ N(loc, cov) <=> xᵢ = loc + sqrt zᵢ, where zᵢ ~ N(0, I).
        def affine_transformation(x):
            return self.loc + sqrt @ x

        return vmap(affine_transformation)(Z)

    def sample(
        self, seed: KeyArray, sample_shape: Tuple[int, ...]
    ):  # pylint: disable=useless-super-delegation
        r"""See `Distribution.sample`."""
        return self._sample_n(
            seed, sample_shape[0]
        )  # TODO this looks weird, why ignore the second entry?

    def kl_divergence(self, other: "GaussianDistribution") -> ScalarFloat:
        return _kl_divergence(self, other)


DistrT = TypeVar("DistrT", bound=tfd.Distribution)


class ReshapedDistribution(tfd.Distribution, Generic[DistrT]):
    def __init__(self, distribution: tfd.Distribution, output_shape: Tuple[int, ...]):
        self._distribution = distribution
        self._output_shape = output_shape

    def mean(self) -> Float[Array, " N ..."]:
        r"""Mean of the base distribution, reshaped to the output shape."""
        return jnp.reshape(self._distribution.mean(), self._output_shape)

    def median(self) -> Float[Array, " N ..."]:
        r"""Median of the base distribution, reshaped to the output shape"""
        return jnp.reshape(self._distribution.median(), self._output_shape)

    def mode(self) -> Float[Array, " N ..."]:
        r"""Mode of the base distribution, reshaped to the output shape"""
        return jnp.reshape(self._distribution.mode(), self._output_shape)

    def covariance(self) -> Float[Array, " N ..."]:
        r"""Covariance of the base distribution, reshaped to the squared output shape"""
        return jnp.reshape(
            self._distribution.covariance(), self._output_shape + self._output_shape
        )

    def variance(self) -> Float[Array, " N ..."]:
        r"""Variances of the base distribution, reshaped to the output shape"""
        return jnp.reshape(self._distribution.variance(), self._output_shape)

    def stddev(self) -> Float[Array, " N ..."]:
        r"""Standard deviations of the base distribution, reshaped to the output shape"""
        return jnp.reshape(self._distribution.stddev(), self._output_shape)

    def entropy(self) -> ScalarFloat:
        r"""Entropy of the base distribution."""
        return self._distribution.entropy()

    def log_prob(
        self, y: Float[Array, " N ..."], mask: Optional[Bool[Array, " N ..."]]
    ) -> ScalarFloat:
        r"""Calculates the log probability."""
        return self._distribution.log_prob(
            y.reshape(-1), mask if mask is None else mask.reshape(-1)
        )

    def sample(
        self, seed: Any, sample_shape: Tuple[int, ...] = ()
    ) -> Float[Array, " n N ..."]:
        r"""Draws samples from the distribution and reshapes them to the output shape."""
        sample = self._distribution.sample(seed, sample_shape)
        return jnp.reshape(sample, sample_shape + self._output_shape)

    def kl_divergence(self, other: "ReshapedDistribution") -> ScalarFloat:
        r"""Calculates the Kullback-Leibler divergence."""
        other_flat = tfd.Distribution(
            loc=other._distribution.loc, scale=other._distribution.scale
        )
        return tfd.kl_divergence(self._distribution, other_flat)

    @property
    def event_shape(self) -> Tuple:
        return self._output_shape


def _check_and_return_dimension(
    q: GaussianDistribution, p: GaussianDistribution
) -> int:
    r"""Checks that the dimensions of the distributions are compatible."""
    if q.event_shape != p.event_shape:
        raise ValueError(
            "Distribution event shapes are not compatible: `q.event_shape ="
            f" {q.event_shape}` and `p.event_shape = {p.event_shape}`. Please check"
            " your mean and covariance shapes."
        )

    return q.event_shape[-1]


def _frobenius_norm_squared(matrix: Float[Array, "N N"]) -> ScalarFloat:
    r"""Calculates the squared Frobenius norm of a matrix."""
    return jnp.sum(jnp.square(matrix))


def _kl_divergence(q: GaussianDistribution, p: GaussianDistribution) -> ScalarFloat:
    r"""KL-divergence between two Gaussians.

    Computes the KL divergence, $`\operatorname{KL}[q\mid\mid p]`$, between two
    multivariate Gaussian distributions $`q(x) = \mathcal{N}(x; \mu_q, \Sigma_q)`$
    and $`p(x) = \mathcal{N}(x; \mu_p, \Sigma_p)`$.

    Args:
        q (GaussianDistribution): A multivariate Gaussian distribution.
        p (GaussianDistribution): A multivariate Gaussian distribution.

    Returns
    -------
        ScalarFloat: The KL divergence between q and p.
    """
    n_dim = _check_and_return_dimension(q, p)

    # Extract q mean and covariance.
    mu_q = q.loc
    sigma_q = q.scale

    # Extract p mean and covariance.
    mu_p = p.loc
    sigma_p = p.scale

    # Find covariance roots.
    sqrt_p = lower_cholesky(sigma_p)
    sqrt_q = lower_cholesky(sigma_q)

    # diff, μp - μq
    diff = mu_p - mu_q

    # trace term, tr[Σp⁻¹ Σq] = tr[(LpLpᵀ)⁻¹(LqLqᵀ)] = tr[(Lp⁻¹Lq)(Lp⁻¹Lq)ᵀ] = (fr[LqLp⁻¹])²
    trace = _frobenius_norm_squared(
        cola.solve(sqrt_p, sqrt_q.to_dense(), Cholesky())
    )  # TODO: Not most efficient, given the `to_dense()` call (e.g., consider diagonal p and q). Need to abstract solving linear operator against another linear operator.

    # Mahalanobis term, (μp - μq)ᵀ Σp⁻¹ (μp - μq) = tr [(μp - μq)ᵀ [LpLpᵀ]⁻¹ (μp - μq)] = (fr[Lp⁻¹(μp - μq)])²
    mahalanobis = jnp.sum(jnp.square(cola.solve(sqrt_p, diff, Cholesky())))

    # KL[q(x)||p(x)] = [ [(μp - μq)ᵀ Σp⁻¹ (μp - μq)] - n - log|Σq| + log|Σp| + tr[Σp⁻¹ Σq] ] / 2
    return (
        mahalanobis
        - n_dim
        - cola.logdet(sigma_q, Cholesky(), Cholesky())
        + cola.logdet(sigma_p, Cholesky(), Cholesky())
        + trace
    ) / 2.0


ReshapedGaussianDistribution = Union[
    GaussianDistribution, ReshapedDistribution[GaussianDistribution]
]
__all__ = [
    "GaussianDistribution",
    "ReshapedDistribution",
    "ReshapedGaussianDistribution",
]
