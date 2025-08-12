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

import jax.numpy as jnp
import jax.random as jr
import pytest

from gpjax.dataset import Dataset
from gpjax.gps import (
    ConjugatePosterior,
    Prior,
)
from gpjax.kernels.orthogonal_additive import (
    OrthogonalAdditiveKernel,
    compute_detailed_sobol_indices,
    compute_sobol_indices,
)
from gpjax.likelihoods import Gaussian
from gpjax.mean_functions import Zero


class TestOrthogonalAdditiveKernel:
    """Test suite for OrthogonalAdditiveKernel."""

    @pytest.fixture
    def kernel_params(self):
        """Standard kernel parameters for testing."""
        return {
            "lengthscales": [1.0, 2.0, 0.5],
            "max_order": 2,
            "input_means": [0.0, 0.0, 0.0],
            "input_scales": [1.0, 1.0, 1.0],
        }

    @pytest.fixture
    def kernel(self, kernel_params):
        """Create a standard OAK kernel for testing."""
        return OrthogonalAdditiveKernel(**kernel_params)

    def test_kernel_initialization(self, kernel_params):
        """Test that kernel initializes correctly."""
        kernel = OrthogonalAdditiveKernel(**kernel_params)

        assert kernel.n_dims == 3
        assert kernel.max_order == 2
        assert kernel.name == "OrthogonalAdditive"

        # Check parameter shapes
        assert kernel.lengthscales.value.shape == (3,)
        assert kernel.input_means.value.shape == (3,)
        assert kernel.input_scales.value.shape == (3,)

        # Default variances should have max_order + 1 elements
        assert len(kernel.variances.value) == 3  # orders 0, 1, 2

    def test_kernel_initialization_defaults(self):
        """Test kernel initialization with default parameters."""
        lengthscales = [1.0, 1.0]
        kernel = OrthogonalAdditiveKernel(lengthscales=lengthscales)

        assert kernel.n_dims == 2
        assert kernel.max_order == 2  # Should default to n_dims

        # Check default values
        assert jnp.allclose(kernel.input_means.value, jnp.zeros(2))
        assert jnp.allclose(kernel.input_scales.value, jnp.ones(2))

    def test_kernel_call_basic(self, kernel):
        """Test basic kernel evaluation."""
        x = jnp.array([1.0, 2.0, 0.5])
        y = jnp.array([1.5, 1.8, 0.3])

        result = kernel(x, y)

        # Result should be a scalar
        assert result.shape == ()
        assert jnp.isfinite(result)

    def test_kernel_call_identical_points(self, kernel):
        """Test kernel evaluation at identical points."""
        x = jnp.array([1.0, 2.0, 0.5])

        result = kernel(x, x)

        # For identical points, result should be positive
        assert result > 0
        assert jnp.isfinite(result)

    def test_kernel_symmetry(self, kernel):
        """Test that kernel is symmetric: k(x,y) = k(y,x)."""
        key = jr.PRNGKey(42)
        x = jr.normal(key, (3,))
        y = jr.normal(jr.split(key)[0], (3,))

        k_xy = kernel(x, y)
        k_yx = kernel(y, x)

        assert jnp.allclose(k_xy, k_yx, rtol=1e-5)

    def test_rbf_kernel_component(self, kernel):
        """Test the RBF kernel component function."""
        x, y = 1.0, 1.5
        lengthscale = 2.0

        result = kernel._rbf_kernel(x, y, lengthscale)

        # Should match manual RBF calculation
        expected = jnp.exp(-0.5 * (x - y) ** 2 / lengthscale**2)
        assert jnp.allclose(result, expected)

    def test_orthogonal_constraint(self, kernel):
        """Test the orthogonal constraint application."""
        x, y = 1.0, 1.5
        dim = 0

        # Base RBF kernel
        k_base = kernel._rbf_kernel(x, y, kernel.lengthscales.value[dim])

        # Constrained kernel
        k_constrained = kernel._apply_orthogonal_constraint(k_base, x, y, dim)

        # Constrained kernel should be different from base kernel
        assert not jnp.allclose(k_base, k_constrained)
        assert jnp.isfinite(k_constrained)

    def test_newton_girard_computation(self, kernel):
        """Test Newton-Girard additive terms computation."""
        # Create simple base kernels
        base_kernels = [0.8, 0.6, 0.4]

        additive_terms = kernel._compute_additive_terms(base_kernels)

        # Should have max_order + 1 terms
        assert len(additive_terms) == min(kernel.max_order + 1, len(base_kernels) + 1)

        # First term should be 1.0 (constant)
        assert jnp.allclose(additive_terms[0], 1.0)

        # Second term should be sum of base kernels
        expected_sum = sum(base_kernels)
        assert jnp.allclose(additive_terms[1], expected_sum)

    def test_newton_girard_empty_kernels(self, kernel):
        """Test Newton-Girard with empty kernel list."""
        base_kernels = []
        additive_terms = kernel._compute_additive_terms(base_kernels)

        # Should return at least the constant term
        assert len(additive_terms) >= 1
        assert jnp.allclose(additive_terms[0], 1.0)

    def test_kernel_matrix_computation(self, kernel):
        """Test kernel gram matrix computation."""
        key = jr.PRNGKey(42)
        X = jr.normal(key, (5, 3))

        # Compute gram matrix
        K = kernel.gram(X)

        # Should be positive definite
        assert K.shape == (5, 5)
        # Check some basic properties
        K_dense = K.to_dense()
        assert jnp.allclose(K_dense, K_dense.T)  # Symmetric
        assert jnp.all(jnp.diag(K_dense) > 0)  # Positive diagonal

    def test_cross_covariance(self, kernel):
        """Test cross-covariance computation."""
        key = jr.PRNGKey(42)
        keys = jr.split(key, 2)
        X1 = jr.normal(keys[0], (3, 3))
        X2 = jr.normal(keys[1], (4, 3))

        K12 = kernel.cross_covariance(X1, X2)

        assert K12.shape == (3, 4)
        assert jnp.all(jnp.isfinite(K12))

    def test_max_order_truncation(self):
        """Test that max_order properly truncates interactions."""
        lengthscales = [1.0, 1.0, 1.0]

        # Create kernel with max_order < n_dims
        kernel = OrthogonalAdditiveKernel(lengthscales=lengthscales, max_order=1)

        assert kernel.max_order == 1
        assert len(kernel.variances.value) == 2  # orders 0, 1 only


class TestSobolIndices:
    """Test suite for Sobol indices computation."""

    @pytest.fixture
    def mock_data(self):
        """Generate mock training data."""
        key = jr.PRNGKey(123)
        X = jr.normal(key, (20, 3))
        y = X[:, 0] + 0.5 * X[:, 1] + 0.1 * jr.normal(jr.split(key)[0], (20,))
        y = y.reshape(-1, 1)  # Make y 2D as required by GPJax
        return X, y

    @pytest.fixture
    def mock_posterior(self, mock_data):
        """Create a mock posterior with OAK kernel."""
        X, y = mock_data

        kernel = OrthogonalAdditiveKernel(lengthscales=[1.0, 1.0, 1.0], max_order=2)
        mean_function = Zero()
        prior = Prior(kernel=kernel, mean_function=mean_function)

        likelihood = Gaussian(num_datapoints=len(y))
        dataset = Dataset(X=X, y=y)

        posterior = ConjugatePosterior(prior=prior, likelihood=likelihood)
        return posterior, dataset

    def test_compute_sobol_indices_basic(self, mock_posterior):
        """Test basic Sobol indices computation."""
        posterior, dataset = mock_posterior

        sobol_indices = compute_sobol_indices(posterior, dataset.X, dataset.y)

        # Should return a dictionary
        assert isinstance(sobol_indices, dict)

        # Should contain expected keys
        expected_keys = {"constant", "main_effects", "order_2_interactions"}
        assert set(sobol_indices.keys()) == expected_keys

        # Values should sum to 1 (approximately)
        total = sum(sobol_indices.values())
        assert jnp.allclose(total, 1.0, rtol=1e-3)

        # All values should be non-negative
        for value in sobol_indices.values():
            assert value >= 0

    def test_compute_detailed_sobol_indices(self, mock_posterior):
        """Test detailed Sobol indices computation."""
        posterior, dataset = mock_posterior

        detailed_indices = compute_detailed_sobol_indices(
            posterior, dataset.X, dataset.y, n_samples=100
        )

        # Should return nested dictionary structure
        assert isinstance(detailed_indices, dict)
        assert "total_variance" in detailed_indices
        assert "main_effects" in detailed_indices
        assert "interactions" in detailed_indices
        assert "orders" in detailed_indices

        # Total variance should be positive
        assert detailed_indices["total_variance"] > 0

        # Main effects should have entries for each feature
        assert len(detailed_indices["main_effects"]) == 3

    def test_sobol_indices_wrong_kernel(self, mock_data):
        """Test Sobol indices with non-OAK kernel."""
        from gpjax.kernels import RBF

        X, y = mock_data

        # Create posterior with regular RBF kernel
        kernel = RBF()
        mean_function = Zero()
        prior = Prior(kernel=kernel, mean_function=mean_function)

        likelihood = Gaussian(num_datapoints=len(y))
        posterior = ConjugatePosterior(prior=prior, likelihood=likelihood)

        # Should raise ValueError
        with pytest.raises(ValueError, match="must use OrthogonalAdditiveKernel"):
            compute_sobol_indices(posterior, X, y)

    def test_sobol_indices_reproducible(self, mock_posterior):
        """Test that Sobol indices computation is reproducible."""
        posterior, dataset = mock_posterior

        indices1 = compute_sobol_indices(posterior, dataset.X, dataset.y)
        indices2 = compute_sobol_indices(posterior, dataset.X, dataset.y)

        # Results should be identical
        for key in indices1:
            assert jnp.allclose(indices1[key], indices2[key])


class TestIntegration:
    """Integration tests for OAK kernel with GP inference."""

    def test_full_gp_workflow(self):
        """Test complete GP workflow with OAK kernel."""
        # Generate synthetic data
        key = jr.PRNGKey(42)
        X = jr.normal(key, (30, 2))
        # True function: additive with interactions
        y_true = X[:, 0] ** 2 + 2 * X[:, 1] + 0.5 * X[:, 0] * X[:, 1]
        y = y_true + 0.1 * jr.normal(jr.split(key)[0], (30,))
        y = y.reshape(-1, 1)  # Make y 2D as required by GPJax

        # Create OAK GP
        kernel = OrthogonalAdditiveKernel(lengthscales=[1.0, 1.0], max_order=2)
        mean_function = Zero()
        prior = Prior(kernel=kernel, mean_function=mean_function)

        likelihood = Gaussian(num_datapoints=len(y))
        dataset = Dataset(X=X, y=y)

        posterior = ConjugatePosterior(prior=prior, likelihood=likelihood)

        # Test prediction
        X_test = jr.normal(jr.split(key)[1], (10, 2))
        pred_dist = posterior(X_test, dataset)

        assert pred_dist.mean.shape == (10,)
        # The covariance can be accessed as either a function or matrix - try both
        try:
            cov_shape = pred_dist.covariance.to_dense().shape
        except AttributeError:
            try:
                cov_shape = pred_dist.covariance().shape
            except (AttributeError, TypeError):
                cov_shape = pred_dist.covariance.shape
        assert cov_shape == (10, 10)
        assert jnp.all(jnp.isfinite(pred_dist.mean))

        # Test Sobol indices
        sobol_indices = compute_sobol_indices(posterior, X, y)
        assert len(sobol_indices) == 3  # constant, main, order_2

        # Main effects should be significant (since we have additive structure)
        assert sobol_indices["main_effects"] > 0.1

    def test_comparison_with_rbf(self):
        """Test that OAK gives reasonable results compared to RBF."""
        from gpjax.kernels import RBF

        # Generate data from additive function
        key = jr.PRNGKey(123)
        X = jr.normal(key, (25, 2))
        y = X[:, 0] + X[:, 1] + 0.1 * jr.normal(jr.split(key)[0], (25,))
        y = y.reshape(-1, 1)  # Make y 2D as required by GPJax

        dataset = Dataset(X=X, y=y)

        # OAK kernel
        oak_kernel = OrthogonalAdditiveKernel(lengthscales=[1.0, 1.0])
        oak_prior = Prior(kernel=oak_kernel, mean_function=Zero())
        oak_likelihood = Gaussian(num_datapoints=len(y))
        oak_posterior = ConjugatePosterior(prior=oak_prior, likelihood=oak_likelihood)

        # RBF kernel
        rbf_kernel = RBF(lengthscale=1.0)
        rbf_prior = Prior(kernel=rbf_kernel, mean_function=Zero())
        rbf_likelihood = Gaussian(num_datapoints=len(y))
        rbf_posterior = ConjugatePosterior(prior=rbf_prior, likelihood=rbf_likelihood)

        # Test predictions
        X_test = jr.normal(jr.split(key)[1], (5, 2))

        oak_pred = oak_posterior(X_test, dataset)
        rbf_pred = rbf_posterior(X_test, dataset)

        # Both should give finite predictions
        assert jnp.all(jnp.isfinite(oak_pred.mean))
        assert jnp.all(jnp.isfinite(rbf_pred.mean))

        # Predictions should be in similar range (not exact test)
        oak_range = jnp.max(oak_pred.mean) - jnp.min(oak_pred.mean)
        rbf_range = jnp.max(rbf_pred.mean) - jnp.min(rbf_pred.mean)

        # Ranges should be of same order of magnitude
        assert oak_range / rbf_range < 10
        assert rbf_range / oak_range < 10
