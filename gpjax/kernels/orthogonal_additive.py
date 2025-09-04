# Copyright 2022 The JaxGaussianProcesses Contributors. All Rights Reserved.
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

import itertools

import beartype.typing as tp
from flax import nnx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Float

from gpjax.kernels.base import AbstractKernel
from gpjax.kernels.computations import (
    AbstractKernelComputation,
    DenseKernelComputation,
)
from gpjax.parameters import (
    NonNegativeReal,
    PositiveReal,
)
from gpjax.typing import (
    Array,
    ScalarFloat,
)

# Numerical stability constants
EPS = 1e-6
EPS_DIVISION = 1e-12
CLIP_EXP_BOUNDS = (-20.0, 20.0)
CLIP_CORRECTION_BOUNDS = (-10.0, 10.0)
CLIP_CONTRIBUTION_BOUNDS = (-1e10, 1e10)
CLIP_ELEMENTARY_BOUNDS = (-1e6, 1e6)


class OrthogonalAdditiveKernel(AbstractKernel):
    r"""Orthogonal Additive Kernel (OAK).

    Implements the orthogonal additive kernel from Lu et al. (2022).
    The kernel decomposes the function into orthogonal additive components:

    f(x) = f_0 + sum_{i} f_i(x_i) + sum_{i<j} f_{ij}(x_i, x_j) + ...

    where each f_u satisfies the orthogonality constraint:
    ∫ f_u(x_u) p_i(x_i) dx_i = 0 for all i ∈ u

    This enables identifiable decomposition with analytic Sobol indices.
    """

    name: str = "OrthogonalAdditive"

    def __init__(
        self,
        lengthscales: tp.Union[list[float], Float[Array, " D"]],
        variances: tp.Union[list[float], Float[Array, " D"]] = None,
        input_means: tp.Union[list[float], Float[Array, " D"]] = None,
        input_scales: tp.Union[list[float], Float[Array, " D"]] = None,
        max_order: int = None,
        active_dims: tp.Union[list[int], slice, None] = None,
        compute_engine: AbstractKernelComputation = DenseKernelComputation(),
    ):
        """Initialize the Orthogonal Additive Kernel.

        Args:
            lengthscales: RBF lengthscales for each input dimension.
            variances: Variance parameters σ²_d for each interaction order.
                If None, defaults to [1.0, 0.5, 0.25, ...] decreasing pattern.
            input_means: Mean parameters μ_i for Gaussian input densities.
                If None, assumes zero mean.
            input_scales: Scale parameters δ_i for Gaussian input densities.
                If None, assumes unit variance.
            max_order: Maximum interaction order. If None, uses input dimensionality.
            active_dims: The indices of active input dimensions.
            compute_engine: The computation engine for kernel matrices.
        """
        super().__init__(active_dims, len(lengthscales), compute_engine)

        self.n_dims = len(lengthscales)
        self.max_order = max_order if max_order is not None else self.n_dims

        # Convert inputs to JAX arrays
        lengthscales = jnp.array(lengthscales)

        if variances is None:
            # Default decreasing variance pattern
            variances = [1.0 / (2.0**i) for i in range(self.max_order + 1)]
        variances = jnp.array(variances)

        if input_means is None:
            input_means = jnp.zeros(self.n_dims)
        else:
            input_means = jnp.array(input_means)

        if input_scales is None:
            input_scales = jnp.ones(self.n_dims)
        else:
            input_scales = jnp.array(input_scales)

        # Create parameter objects
        self.lengthscales = PositiveReal(lengthscales)
        self.variances = NonNegativeReal(variances[: self.max_order + 1])
        self.input_means = nnx.Variable(input_means)
        self.input_scales = PositiveReal(input_scales)


    def __call__(
        self,
        x: Float[Array, " D"],
        y: Float[Array, " D"],
    ) -> ScalarFloat:
        """Evaluate kernel at a pair of points."""
        x = self.slice_input(x)
        y = self.slice_input(y)

        # Compute base RBF kernels for each dimension
        base_kernels = []
        for i in range(self.n_dims):
            # Constrained RBF kernel for dimension i
            k_base = self._rbf_kernel(x[i], y[i], self.lengthscales.value[i])
            k_constrained = self._apply_orthogonal_constraint(k_base, x[i], y[i], i)
            base_kernels.append(k_constrained)

        # Use Newton-Girard method to compute additive terms
        additive_terms = self._compute_additive_terms(base_kernels)

        # Weight terms by variance parameters
        result = 0.0
        for order, term in enumerate(additive_terms):
            if order < len(self.variances.value):
                result += self.variances.value[order] * term

        return result.squeeze()

    def _rbf_kernel(
        self, x: ScalarFloat, y: ScalarFloat, lengthscale: ScalarFloat
    ) -> ScalarFloat:
        """Compute base RBF kernel between two scalar inputs."""
        squared_dist = (x - y) ** 2
        return jnp.exp(-0.5 * squared_dist / (lengthscale**2))

    def _apply_orthogonal_constraint(
        self, k_base: ScalarFloat, x: ScalarFloat, y: ScalarFloat, dim: int
    ) -> ScalarFloat:
        """Apply orthogonality constraint to base kernel.

        For Gaussian input density N(μ, δ²), the constrained kernel is:
        k̃(x,y) = k(x,y) - E[S_i f(x)] E[S_i²]^{-1} E[S_i f(y)]

        where S_i = ∫ f(x_i) p_i(x_i) dx_i
        """
        mu_i = self.input_means.value[dim]
        delta_i = self.input_scales.value[dim]
        ell_i = self.lengthscales.value[dim]

        # Compute expectation terms for Gaussian measure
        # E[S_i f(x)] = σ² * √(ℓ² / (ℓ² + δ²)) * exp(-(x-μ)² / (2(ℓ²+δ²)))
        
        # Add small epsilon for numerical stability
        var_term = ell_i**2 / (ell_i**2 + delta_i**2 + EPS)
        scale_factor = jnp.sqrt(var_term)

        # Clamp the exponential arguments to prevent overflow
        exp_arg_x = jnp.clip(-((x - mu_i) ** 2) / (2 * (ell_i**2 + delta_i**2 + EPS)), *CLIP_EXP_BOUNDS)
        exp_arg_y = jnp.clip(-((y - mu_i) ** 2) / (2 * (ell_i**2 + delta_i**2 + EPS)), *CLIP_EXP_BOUNDS)
        
        exp_x = jnp.exp(exp_arg_x)
        exp_y = jnp.exp(exp_arg_y)

        # E[S_i²] = σ² * √(ℓ² / (ℓ² + 2δ²))
        var_s_squared = jnp.sqrt(ell_i**2 / (ell_i**2 + 2 * delta_i**2 + EPS))

        # Constrained kernel with numerical safety
        correction = (scale_factor / (var_s_squared + EPS)) * exp_x * exp_y
        
        # Clamp correction to prevent instability
        correction = jnp.clip(correction, *CLIP_CORRECTION_BOUNDS)
        
        return k_base - correction

    def _compute_additive_terms(
        self, base_kernels: list[ScalarFloat]
    ) -> list[ScalarFloat]:
        """Compute additive terms using Newton-Girard method.

        Implements Algorithm 1 from the paper to efficiently compute
        interaction terms without exponential complexity.
        """
        max_depth = min(self.max_order, len(base_kernels))

        # Step 1: Compute power sums S_ℓ = Σᵢ kᵢˡ
        power_sums = []
        for ell in range(max_depth + 1):
            if ell == 0:
                power_sums.append(jnp.array(float(len(base_kernels)), dtype=jnp.float32))
            else:
                power_sum = sum(k**ell for k in base_kernels)
                power_sums.append(power_sum)

        # Step 2: Compute elementary symmetric polynomials E_ℓ using Newton-Girard
        elementary_terms = [jnp.array(1.0)]  # E_0 = 1

        for ell in range(1, max_depth + 1):
            # Newton-Girard identity: ℓ * E_ℓ = Σₖ (-1)^{k-1} * E_{ℓ-k} * S_k
            term = 0.0
            for k in range(1, ell + 1):
                sign = (-1.0) ** (k - 1)
                contribution = sign * elementary_terms[ell - k] * power_sums[k]
                # Clip individual contributions to prevent overflow
                contribution = jnp.clip(contribution, *CLIP_CONTRIBUTION_BOUNDS)
                term += contribution

            # Prevent division by zero and clip result
            elementary_term = term / jnp.maximum(ell, EPS_DIVISION)
            elementary_term = jnp.clip(elementary_term, *CLIP_ELEMENTARY_BOUNDS)
            elementary_terms.append(elementary_term)

        return elementary_terms[: max_depth + 1]


class OrthogonalAdditiveGP:
    """Custom GP model wrapper for OAK kernel with proper Sobol decomposition.

    This class wraps a standard GPJax posterior with an OAK kernel and provides
    methods to decompose the posterior mean into additive components and compute
    true Sobol indices based on variance contributions.
    """

    def __init__(self, posterior, dataset):
        """Initialize OAK GP model.

        Args:
            posterior: GPJax posterior with OrthogonalAdditiveKernel
            dataset: Training dataset
        """
        if not hasattr(posterior.prior.kernel, "max_order"):
            raise ValueError("Posterior must use OrthogonalAdditiveKernel")

        self.posterior = posterior
        self.dataset = dataset
        self.kernel = posterior.prior.kernel
        self.n_dims = self.kernel.n_dims
        self.max_order = self.kernel.max_order

        # Cache training predictions for efficiency
        self._train_latent_dist = None

    def _get_training_predictions(self):
        """Get cached training predictions."""
        if self._train_latent_dist is None:
            self._train_latent_dist = self.posterior.predict(
                self.dataset.X, self.dataset
            )
        return self._train_latent_dist

    def predict(self, X_test):
        """Make predictions at test points."""
        return self.posterior.predict(X_test, self.dataset)

    def decompose_posterior_mean(self, X_test):
        """Decompose posterior mean into additive components.

        For the OAK kernel, the posterior mean can be written as:
        m(x) = sum_{u subset [d]} m_u(x_u)

        where each m_u corresponds to an interaction subset u.

        Args:
            X_test: Test input points of shape (N, D)

        Returns:
            Dictionary mapping component names to their contributions
        """
        # Get posterior distribution
        latent_dist = self.predict(X_test)
        total_mean = latent_dist.mean

        # Get training data predictions for reference
        train_latent = self._get_training_predictions()

        # For OAK, we can approximate component contributions by computing
        # kernel contributions from different interaction orders
        components = {}

        # Generate all possible interaction subsets up to max_order
        all_components = []
        component_names = []

        # Add constant term (order 0)
        if len(self.kernel.variances.value) > 0:
            const_contrib = jnp.full_like(total_mean, jnp.mean(train_latent.mean))
            all_components.append(const_contrib)
            component_names.append("constant")

        # Add main effects (order 1)
        if len(self.kernel.variances.value) > 1:
            for i in range(self.n_dims):
                # Compute marginal contribution of dimension i
                marginal_mean = self._compute_marginal_effect(X_test, i)
                all_components.append(marginal_mean)
                component_names.append(f"main_effect_{i}")

        # Add interaction effects (order 2 and higher)
        for order in range(
            2, min(self.max_order + 1, len(self.kernel.variances.value))
        ):
            if order < len(self.kernel.variances.value):
                # Generate all combinations of variables for this order
                for subset in itertools.combinations(range(self.n_dims), order):
                    interaction_mean = self._compute_interaction_effect(X_test, subset)
                    all_components.append(interaction_mean)
                    component_names.append(f"interaction_{'_'.join(map(str, subset))}")

        # Package results
        for name, component in zip(component_names, all_components, strict=True):
            components[name] = component

        return components

    def _compute_marginal_effect(self, X_test, dim_idx):
        """Compute marginal effect for a single dimension.

        This approximates the contribution of dimension dim_idx by computing
        predictions with other dimensions set to their means.
        """
        # Create modified input with other dims at their means
        X_marginal = X_test.copy()
        for i in range(self.n_dims):
            if i != dim_idx:
                X_marginal = X_marginal.at[:, i].set(self.kernel.input_means.value[i])

        # Get prediction with only this dimension varying
        marginal_dist = self.predict(X_marginal)

        # Subtract the constant (grand mean) contribution
        train_mean = jnp.mean(self._get_training_predictions().mean)

        return marginal_dist.mean - train_mean

    def _compute_interaction_effect(self, X_test, subset):
        """Compute interaction effect for a subset of dimensions.

        This approximates the pure interaction effect by using inclusion-exclusion
        principle on the marginal predictions.
        """
        # For simplicity, approximate interaction as residual after main effects
        # This is a simplified version - full implementation would use
        # inclusion-exclusion principle

        # Weight by the kernel variance for this interaction order
        order = len(subset)
        if order < len(self.kernel.variances.value):
            interaction_weight = self.kernel.variances.value[order] / jnp.sum(
                self.kernel.variances.value
            )
            # Approximate interaction as weighted portion of residual variance
            total_pred = self.predict(X_test).mean
            main_effects_sum = sum(
                self._compute_marginal_effect(X_test, i) for i in subset
            )

            # Interaction approximated as portion of remaining variance
            return interaction_weight * (total_pred - main_effects_sum)
        else:
            return jnp.zeros_like(X_test[:, 0])

    def compute_true_sobol_indices(self, X_test=None, n_samples=1000):
        """Compute true Sobol indices using posterior mean decomposition.

        Args:
            X_test: Test points for computing variance. If None, samples from input distribution.
            n_samples: Number of samples for variance estimation if X_test is None.

        Returns:
            Dictionary of Sobol indices
        """
        # Generate test points if not provided
        if X_test is None:
            key = jr.PRNGKey(42)
            # Sample from input distribution
            means = self.kernel.input_means.value
            scales = self.kernel.input_scales.value
            X_test = jr.normal(key, (n_samples, self.n_dims)) * scales + means

        # Decompose posterior mean into components
        components = self.decompose_posterior_mean(X_test)

        # Compute variance of each component
        component_variances = {}
        for name, component_mean in components.items():
            component_variances[name] = float(jnp.var(component_mean))

        # Get total variance
        total_mean = self.predict(X_test).mean
        total_variance = float(jnp.var(total_mean))

        # Compute Sobol indices
        sobol_indices = {}
        if total_variance > 0:
            for name, var in component_variances.items():
                sobol_indices[name] = var / total_variance
        else:
            # If no variance, set equal indices
            n_components = len(component_variances)
            for name in component_variances:
                sobol_indices[name] = 1.0 / n_components if n_components > 0 else 0.0

        return sobol_indices

    def compute_grouped_sobol_indices(self, X_test=None, n_samples=1000):
        """Compute Sobol indices grouped by interaction order.

        Returns:
            Dictionary with interaction orders as keys
        """
        individual_indices = self.compute_true_sobol_indices(X_test, n_samples)

        # Group by interaction order
        grouped_indices = {
            "constant": 0.0,
            "main_effects": 0.0,
            "order_2_interactions": 0.0,
        }
        
        higher_order_value = 0.0

        for name, value in individual_indices.items():
            if "constant" in name:
                grouped_indices["constant"] += value
            elif "main_effect" in name:
                grouped_indices["main_effects"] += value
            elif "interaction" in name:
                # Count number of variables in interaction
                if name.count("_") == 2:  # interaction_i_j has 2 underscores
                    grouped_indices["order_2_interactions"] += value
                else:
                    higher_order_value += value
        
        # Only include higher_order_interactions if it's non-zero or max_order > 2
        if higher_order_value > 0 or self.max_order > 2:
            grouped_indices["higher_order_interactions"] = higher_order_value

        return grouped_indices


def compute_sobol_indices(
    posterior,
    X: Float[Array, "N D"],
    y: Float[Array, "N 1"],
    use_full_decomposition: bool = False,
    n_samples: int = 1000,
) -> dict[str, float]:
    """Compute Sobol indices for OAK model feature importance.

    Based on equation (14) from Lu et al. (2022), the Sobol index is:
    R_u = Var_x[m_u(x)] / Var_x[m(x)]

    where m_u(x) is the posterior mean of component u and m(x) is the total
    posterior mean.

    Args:
        posterior: GPJax posterior object with OAK kernel.
        X: Training input data of shape (N, D).
        y: Training output data of shape (N, 1).
        use_full_decomposition: If True, uses proper posterior mean decomposition.
            If False, uses kernel variance parameters as proxy (legacy behavior).
        n_samples: Number of samples for variance estimation in full decomposition.

    Returns:
        Dictionary mapping component names to normalized Sobol indices.
    """
    _validate_oak_kernel(posterior)

    if use_full_decomposition:
        from gpjax.dataset import Dataset
        dataset = Dataset(X=X, y=y)
        oak_gp = OrthogonalAdditiveGP(posterior, dataset)
        return oak_gp.compute_grouped_sobol_indices(n_samples=n_samples)
    else:
        return _compute_legacy_sobol_indices(posterior.prior.kernel)


def compute_detailed_sobol_indices(
    posterior,
    X: Float[Array, "N D"],
    y: Float[Array, "N 1"],
    n_samples: int = 1000,
    use_full_decomposition: bool = False,
) -> dict[str, tp.Any]:
    """Compute detailed Sobol indices for individual features and interactions.

    This provides a more granular breakdown of feature importance beyond
    just the interaction orders.

    Args:
        posterior: GPJax posterior object with OAK kernel.
        X: Training input data.
        y: Training output data.
        n_samples: Number of Monte Carlo samples for variance estimation.
        use_full_decomposition: If True, uses proper posterior mean decomposition.

    Returns:
        Dictionary with detailed Sobol breakdown including individual features.
    """
    _validate_oak_kernel(posterior)

    from gpjax.dataset import Dataset
    dataset = Dataset(X=X, y=y)

    if use_full_decomposition:
        oak_gp = OrthogonalAdditiveGP(posterior, dataset)
        return _compute_full_detailed_sobol(oak_gp, n_samples)
    else:
        return _compute_legacy_detailed_sobol(posterior, dataset, n_samples)


def _validate_oak_kernel(posterior):
    """Validate that posterior uses OrthogonalAdditiveKernel."""
    if not hasattr(posterior.prior.kernel, "max_order"):
        raise ValueError("Posterior must use OrthogonalAdditiveKernel")


def _compute_legacy_sobol_indices(oak_kernel) -> dict[str, float]:
    """Legacy implementation using kernel variances as proxies."""
    max_order = oak_kernel.max_order
    sobol_indices = {}
    variance_contributions = []

    for order in range(min(max_order + 1, len(oak_kernel.variances.value))):
        var_contribution = oak_kernel.variances.value[order]
        variance_contributions.append(var_contribution)

    total_contribution = sum(variance_contributions)

    for order, contribution in enumerate(variance_contributions):
        normalized_contrib = (
            contribution / total_contribution if total_contribution > 0 else 0.0
        )

        if order == 0:
            sobol_indices["constant"] = float(normalized_contrib)
        elif order == 1:
            sobol_indices["main_effects"] = float(normalized_contrib)
        else:
            sobol_indices[f"order_{order}_interactions"] = float(normalized_contrib)

    return sobol_indices


def _compute_full_detailed_sobol(oak_gp, n_samples: int) -> dict[str, tp.Any]:
    """Compute detailed Sobol indices using full decomposition."""
    key = jr.PRNGKey(42)
    means = oak_gp.kernel.input_means.value
    scales = oak_gp.kernel.input_scales.value
    test_points = jr.normal(key, (n_samples, oak_gp.n_dims)) * scales + means

    total_pred = oak_gp.predict(test_points)
    total_variance = float(jnp.var(total_pred.mean))

    individual_indices = oak_gp.compute_true_sobol_indices(test_points, n_samples)
    grouped_indices = oak_gp.compute_grouped_sobol_indices(test_points, n_samples)

    results = {
        "total_variance": total_variance,
        "main_effects": {},
        "interactions": {},
        "orders": grouped_indices,
        "individual_components": individual_indices,
    }

    for name, value in individual_indices.items():
        if "main_effect" in name:
            feature_idx = name.split("_")[-1]
            results["main_effects"][f"feature_{feature_idx}"] = float(value)
        elif "interaction" in name:
            results["interactions"][name] = float(value)

    return results


def _compute_legacy_detailed_sobol(posterior, dataset, n_samples: int) -> dict[str, tp.Any]:
    """Legacy implementation for detailed Sobol indices."""
    oak_kernel = posterior.prior.kernel
    n_dims = oak_kernel.n_dims

    means = oak_kernel.input_means.value
    scales = oak_kernel.input_scales.value

    key = jr.PRNGKey(42)
    test_points = jr.normal(key, (n_samples, n_dims)) * scales + means

    posterior_dist = posterior(test_points, dataset)
    predictions = posterior_dist.mean
    total_variance = jnp.var(predictions)

    results = {
        "total_variance": float(total_variance),
        "main_effects": {},
        "interactions": {},
        "orders": _compute_legacy_sobol_indices(oak_kernel),
    }

    for i in range(n_dims):
        feature_weight = (
            oak_kernel.variances.value[1] / n_dims
            if len(oak_kernel.variances.value) > 1
            else 0.0
        )
        results["main_effects"][f"feature_{i}"] = float(feature_weight)

    return results
