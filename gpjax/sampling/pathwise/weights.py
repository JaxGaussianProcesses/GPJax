# Copyright 2024 The GPJax Contributors. All Rights Reserved.
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
"""Weight samplers for pathwise sampling.

This module implements weight samplers that generate the coefficients θᵢ
for finite feature approximations in pathwise sampling. Different samplers
implement different strategies for drawing weights appropriate for prior
and posterior sampling.

The weight sampling strategies follow Wilson et al. (2020) and extract
logic from existing GPJax implementations.

References:
    Wilson, J., Borovitskiy, V., Terenin, A., Mostowsky, P., & Deisenroth, M. (2020).
    Efficiently sampling functions from Gaussian process posteriors.
    International Conference on Machine Learning.
"""

from __future__ import annotations

import beartype.typing as tp
import jax.numpy as jnp
import jax.random as jr
from beartype import beartype
from jaxtyping import (
    Array,
    Float,
    jaxtyped,
)

from gpjax.dataset import Dataset
from gpjax.gps import AbstractPrior
from gpjax.likelihoods import Gaussian
from gpjax.linalg import Dense
from gpjax.linalg import solve
from gpjax.linalg.utils import add_jitter
from gpjax.sampling.pathwise.base import AbstractWeightSampler
from gpjax.sampling.pathwise.features import AbstractFeatureGenerator
from gpjax.typing import KeyArray


class GaussianWeightSampler(AbstractWeightSampler):
    """Standard Gaussian weight sampler for prior sampling.
    
    Samples weights from a standard Gaussian distribution, appropriate for
    prior sampling where weights are i.i.d. N(0, 1). This replicates the
    weight sampling from Prior.sample_approx().
    
    For Fourier features, the weights θᵢ ~ N(0, 1) give the correct prior
    covariance when combined with properly scaled features.
    """

    @jaxtyped(typechecker=beartype)
    def sample_weights(
        self,
        num_samples: int,
        num_features: int,
        key: KeyArray,
    ) -> Float[Array, "B F"]:
        """Sample standard Gaussian weights.
        
        Args:
            num_samples: Number of function samples (batch size).
            num_features: Number of features needing weights.
            key: Random key for sampling.
            
        Returns:
            Weights with shape (B, F) where each entry ~ N(0, 1).
        """
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if num_features <= 0:
            raise ValueError("num_features must be positive")
            
        return jr.normal(key, shape=(num_samples, num_features))


class PosteriorWeightSampler(AbstractWeightSampler):
    """Posterior-adapted weight sampler for accurate posterior sampling.
    
    Computes weights that properly account for the conditioning on training
    data, following Wilson et al. (2020). This extracts and generalizes the
    weight computation from ConjugatePosterior.sample_approx().
    
    For composite features [φ_fourier, φ_canonical], the weights are computed
    to ensure the correct posterior covariance structure.
    """

    def __init__(
        self,
        prior: AbstractPrior,
        likelihood: Gaussian,
        train_data: Dataset,
        feature_generator: AbstractFeatureGenerator,
    ):
        """Initialize the posterior weight sampler.
        
        Args:
            prior: The Gaussian process prior.
            likelihood: Gaussian likelihood (for observation noise).
            train_data: Training data for conditioning.
            feature_generator: Generator for computing features at training inputs.
        """
        self.prior = prior
        self.likelihood = likelihood
        self.train_data = train_data
        self.feature_generator = feature_generator

    @jaxtyped(typechecker=beartype)
    def sample_weights(
        self,
        num_samples: int,
        num_features: int,
        key: KeyArray,
    ) -> Float[Array, "B F"]:
        """Sample posterior-adapted weights.
        
        Implements the weight computation from ConjugatePosterior.sample_approx(),
        which computes canonical weights via solving a linear system and
        keeps Fourier weights as standard Gaussian.
        
        CRITICAL FIX: Uses the same key for Fourier weights and noise sampling
        to match the original implementation's behavior exactly.
        
        Args:
            num_samples: Number of function samples (batch size).
            num_features: Number of features needing weights.
            key: Random key for sampling.
            
        Returns:
            Weights with shape (B, F) where F includes both Fourier and canonical features.
        """
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
            
        # CRITICAL FIX: Use the same key for both Fourier weights and noise
        # to match the original implementation exactly (lines 578 and 583 in gps.py)
        
        # Sample standard Gaussian weights for Fourier features
        # This assumes num_features includes both Fourier and canonical
        # We need to determine the split - for now, assume it's from a composite generator
        if hasattr(self.feature_generator, 'fourier_generator'):
            num_fourier = self.feature_generator.fourier_generator.num_features
        else:
            # Fallback: assume all features are Fourier (prior sampling case)
            num_fourier = num_features
            
        # Use the same key for Fourier weights as original implementation
        fourier_weights = jr.normal(key, shape=(num_samples, num_fourier))
        
        # If we have canonical features, compute their weights via linear solve
        if hasattr(self.feature_generator, 'canonical_generator'):
            num_canonical = self.feature_generator.canonical_generator.num_features
            canonical_weights = self._compute_canonical_weights(
                fourier_weights, key, num_samples  # Use same key for noise too!
            )
            # Concatenate weights
            weights = jnp.concatenate([fourier_weights, canonical_weights], axis=-1)
        else:
            # Only Fourier features
            weights = fourier_weights
            
        return weights

    def _compute_canonical_weights(
        self,
        fourier_weights: Float[Array, "B F_fourier"],
        key: KeyArray,
        num_samples: int,
    ) -> Float[Array, "B F_canonical"]:
        """Compute canonical feature weights via linear solve.
        
        This replicates the canonical weight computation from the existing
        ConjugatePosterior.sample_approx() implementation.
        
        Args:
            fourier_weights: Already sampled Fourier weights.
            key: Random key for noise sampling.
            num_samples: Number of samples.
            
        Returns:
            Canonical weights computed via solving the linear system.
        """
        # Get observation variance
        obs_var = self.likelihood.obs_stddev.value**2
        
        # Compute covariance matrix at training inputs
        Kxx = self.prior.kernel.gram(self.train_data.X)
        Sigma = Dense(add_jitter(Kxx.to_dense(), obs_var + self.prior.jitter))
        
        # Sample observation noise
        eps = jnp.sqrt(obs_var) * jr.normal(key, shape=(self.train_data.n, num_samples))
        
        # Center training targets
        y_centered = self.train_data.y - self.prior.mean_function(self.train_data.X)
        
        # Compute Fourier features at training inputs
        fourier_features = self.feature_generator.fourier_generator.generate_features(
            self.train_data.X, key
        )
        
        # Compute the right-hand side for the linear solve
        fourier_contribution = jnp.inner(fourier_features, fourier_weights)
        rhs = y_centered + eps - fourier_contribution
        
        # Solve for canonical weights: Sigma * weights = rhs
        canonical_weights = solve(Sigma, rhs)  # Shape: (N, B)
        
        # Transpose to match expected output shape (B, N)
        return canonical_weights.T


class CompositeWeightSampler(AbstractWeightSampler):
    """Composite weight sampler for handling mixed feature types.
    
    Delegates weight sampling to appropriate samplers based on feature type.
    This provides a unified interface for complex sampling scenarios while
    maintaining the correct sampling strategy for each feature type.
    """

    def __init__(
        self,
        fourier_sampler: AbstractWeightSampler,
        canonical_sampler: tp.Optional[AbstractWeightSampler] = None,
        num_fourier_features: int = 0,
    ):
        """Initialize the composite weight sampler.
        
        Args:
            fourier_sampler: Sampler for Fourier feature weights.
            canonical_sampler: Optional sampler for canonical weights.
            num_fourier_features: Number of Fourier features (for splitting).
        """
        self.fourier_sampler = fourier_sampler
        self.canonical_sampler = canonical_sampler
        self.num_fourier_features = num_fourier_features

    @jaxtyped(typechecker=beartype)
    def sample_weights(
        self,
        num_samples: int,
        num_features: int,
        key: KeyArray,
    ) -> Float[Array, "B F"]:
        """Sample weights using appropriate samplers.
        
        Args:
            num_samples: Number of function samples.
            num_features: Total number of features.
            key: Random key for sampling.
            
        Returns:
            Combined weights for all features.
        """
        # Split keys for different samplers
        fourier_key, canonical_key = jr.split(key)
        
        # Sample Fourier weights
        fourier_weights = self.fourier_sampler.sample_weights(
            num_samples, self.num_fourier_features, fourier_key
        )
        
        if self.canonical_sampler is not None:
            num_canonical = num_features - self.num_fourier_features
            canonical_weights = self.canonical_sampler.sample_weights(
                num_samples, num_canonical, canonical_key
            )
            return jnp.concatenate([fourier_weights, canonical_weights], axis=-1)
        else:
            return fourier_weights


def create_gaussian_sampler() -> GaussianWeightSampler:
    """Factory function for creating Gaussian weight samplers.
    
    Returns:
        Configured Gaussian weight sampler for prior sampling.
    """
    return GaussianWeightSampler()


def create_posterior_sampler(
    prior: AbstractPrior,
    likelihood: Gaussian,
    train_data: Dataset,
    feature_generator: AbstractFeatureGenerator,
) -> PosteriorWeightSampler:
    """Factory function for creating posterior weight samplers.
    
    Args:
        prior: The Gaussian process prior.
        likelihood: Gaussian likelihood.
        train_data: Training data.
        feature_generator: Feature generator for computing features at training points.
        
    Returns:
        Configured posterior weight sampler.
    """
    return PosteriorWeightSampler(prior, likelihood, train_data, feature_generator)


__all__ = [
    "GaussianWeightSampler",
    "PosteriorWeightSampler",
    "CompositeWeightSampler",
    "create_gaussian_sampler",
    "create_posterior_sampler",
]