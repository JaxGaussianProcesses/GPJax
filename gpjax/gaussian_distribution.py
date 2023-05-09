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
    Optional,
    Tuple,
)
from jax import vmap
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Float
import tensorflow_probability.substrates.jax as tfp

from gpjax.linops import (
    IdentityLinearOperator,
    LinearOperator,
)
from gpjax.typing import (
    Array,
    KeyArray,
    ScalarFloat,
)

tfd = tfp.distributions


def _check_loc_scale(loc: Optional[Any], scale: Optional[Any]) -> None:
    r"""Checks that the inputs are correct."""
    if loc is None and scale is None:
        raise ValueError("At least one of `loc` or `scale` must be specified.")

    if loc is not None and loc.ndim < 1:
        raise ValueError("The parameter `loc` must have at least one dimension.")

    if scale is not None and scale.ndim < 2:
        raise ValueError(
            "The `scale` must have at least two dimensions, but "
            f"`scale.shape = {scale.shape}`."
        )

    if scale is not None and not isinstance(scale, LinearOperator):
        raise ValueError(
            f"scale must be a LinearOperator or a JAX array, but got {type(scale)}"
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
            scale = IdentityLinearOperator(num_dims)

        self.loc = loc
        self.scale = scale

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
        return self.scale.diagonal()

    def stddev(self) -> Float[Array, " N"]:
        r"""Calculates the standard deviation."""
        return jnp.sqrt(self.scale.diagonal())

    @property
    def event_shape(self) -> Tuple:
        r"""Returns the event shape."""
        return self.loc.shape[-1:]

    def entropy(self) -> ScalarFloat:
        r"""Calculates the entropy of the distribution."""
        return 0.5 * (
            self.event_shape[0] * (1.0 + jnp.log(2.0 * jnp.pi)) + self.scale.log_det()
        )

    def log_prob(self, y: Float[Array, " N"]) -> ScalarFloat:
        r"""Calculates the log pdf of the multivariate Gaussian.

        Args:
            y (Float[Array, " N"]): The value to calculate the log probability of.

        Returns
        -------
            ScalarFloat: The log probability of the value.
        """
        mu = self.loc
        sigma = self.scale
        n = mu.shape[-1]

        # diff, y - µ
        diff = y - mu

        # compute the pdf, -1/2[ n log(2π) + log|Σ| + (y - µ)ᵀΣ⁻¹(y - µ) ]
        return -0.5 * (
            n * jnp.log(2.0 * jnp.pi) + sigma.log_det() + diff.T @ sigma.solve(diff)
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
        sqrt = self.scale.to_root()

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
    sqrt_p = sigma_p.to_root()
    sqrt_q = sigma_q.to_root()

    # diff, μp - μq
    diff = mu_p - mu_q

    # trace term, tr[Σp⁻¹ Σq] = tr[(LpLpᵀ)⁻¹(LqLqᵀ)] = tr[(Lp⁻¹Lq)(Lp⁻¹Lq)ᵀ] = (fr[LqLp⁻¹])²
    trace = _frobenius_norm_squared(
        sqrt_p.solve(sqrt_q.to_dense())
    )  # TODO: Not most efficient, given the `to_dense()` call (e.g., consider diagonal p and q). Need to abstract solving linear operator against another linear operator.

    # Mahalanobis term, (μp - μq)ᵀ Σp⁻¹ (μp - μq) = tr [(μp - μq)ᵀ [LpLpᵀ]⁻¹ (μp - μq)] = (fr[Lp⁻¹(μp - μq)])²
    mahalanobis = jnp.sum(
        jnp.square(sqrt_p.solve(diff))
    )  # TODO: Need to improve this. Perhaps add a Mahalanobis method to ``LinearOperator``s.

    # KL[q(x)||p(x)] = [ [(μp - μq)ᵀ Σp⁻¹ (μp - μq)] - n - log|Σq| + log|Σp| + tr[Σp⁻¹ Σq] ] / 2
    return (mahalanobis - n_dim - sigma_q.log_det() + sigma_p.log_det() + trace) / 2.0


__all__ = [
    "GaussianDistribution",
]
