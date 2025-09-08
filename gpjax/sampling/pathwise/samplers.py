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
"""Concrete pathwise sampler implementations.

This module implements the main pathwise sampler classes that replicate and
extend the functionality of Prior.sample_approx() and ConjugatePosterior.sample_approx().
The samplers follow Wilson et al. (2020) and provide a modular, extensible
interface for pathwise sampling.

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
from gpjax.gps import (
    AbstractPrior,
    ConjugatePosterior,
)
from gpjax.likelihoods import Gaussian
from gpjax.sampling.pathwise.base import AbstractPathwiseSampler
from gpjax.sampling.pathwise.features import (
    FourierFeatureGenerator,
    CompositeFeatureGenerator,
    create_fourier_generator,
    create_composite_generator,
)
from gpjax.sampling.pathwise.weights import (
    GaussianWeightSampler,
    PosteriorWeightSampler,
    create_gaussian_sampler,
    create_posterior_sampler,
)
from gpjax.typing import (
    FunctionalSample,
    KeyArray,
)


class PriorPathwiseSampler(AbstractPathwiseSampler):
    """Pathwise sampler for Gaussian process priors.
    
    Replicates and extends the functionality of Prior.sample_approx() using
    the modular pathwise sampling framework. Uses Fourier features with
    Gaussian weights to approximate prior samples.
    
    The approximation takes the form:
        f̂(x) = μ(x) + Σᵢ φᵢ(x)θᵢ
    
    where μ(x) is the mean function, φᵢ(x) are Fourier features, and θᵢ ~ N(0,1).
    """

    def __init__(
        self,
        prior: AbstractPrior,
        num_features: int = 100,
        jitter: float = 1e-6,
    ):
        """Initialize the prior pathwise sampler.
        
        Args:
            prior: The Gaussian process prior.
            num_features: Number of Fourier basis functions.
            jitter: Small value for numerical stability.
        """
        # Will initialize feature and weight generators in sample()
        # since they need a random key
        super().__init__(
            prior=prior,
            feature_generator=None,  # Set during sampling
            weight_sampler=create_gaussian_sampler(),
            jitter=jitter,
        )
        self.num_fourier_features = num_features

    @jaxtyped(typechecker=beartype)
    def sample(
        self,
        num_samples: int,
        key: KeyArray,
        **kwargs,
    ) -> FunctionalSample:
        """Generate approximate samples from the GP prior.
        
        This method replicates the logic from Prior.sample_approx() but uses
        the modular framework for extensibility.
        
        Args:
            num_samples: Number of function samples to generate.
            key: Random key for sampling.
            **kwargs: Additional arguments (unused for prior sampling).
            
        Returns:
            A function that evaluates prior samples at test inputs.
        """
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
            
        # Split keys for feature generation and weight sampling
        feature_key, weight_key = jr.split(key)
        
        # Create feature generator (needs key for frequency sampling)
        self.feature_generator = create_fourier_generator(
            self.prior, self.num_fourier_features, feature_key
        )
        
        # Pre-initialize RFF kernel for common dimensions to avoid tracer leaks
        # This ensures the RFF kernel is created before any JIT compilation
        for dims in [1, 2, 3]:  # Common dimensions
            try:
                self.feature_generator.precompute_for_dimensions(dims)
            except:
                pass  # Ignore if precompute is not available
        
        # Sample weights for Fourier features
        weights = self.weight_sampler.sample_weights(
            num_samples, self.feature_generator.num_features, weight_key
        )

        def sample_fn(test_inputs: Float[Array, "N D"]) -> Float[Array, "N B"]:
            """Evaluate prior samples at test inputs.
            
            Args:
                test_inputs: Input locations for evaluation.
                
            Returns:
                Sample evaluations with shape (N, B).
            """
            # Generate features at test inputs
            features = self.feature_generator.generate_features(test_inputs, feature_key)
            
            # Compute weighted feature sum
            weighted_features = jnp.inner(features, weights)  # (N, B)
            
            # Add mean function
            mean_values = self.prior.mean_function(test_inputs)  # (N, 1)
            
            return mean_values + weighted_features
        
        return sample_fn


class PosteriorPathwiseSampler(AbstractPathwiseSampler):
    """Pathwise sampler for Gaussian process posteriors.
    
    Replicates and extends the functionality of ConjugatePosterior.sample_approx()
    using the modular pathwise sampling framework. Uses composite features
    (Fourier + canonical) with posterior-adapted weights.
    
    The approximation takes the form:
        f̂(x) = μ(x) + Σᵢ φᵢ(x)θᵢ + Σⱼ k(x, xⱼ)vⱼ
    
    where φᵢ(x) are Fourier features, k(x, xⱼ) are canonical features,
    θᵢ ~ N(0,1), and vⱼ are computed via solving a linear system.
    """

    def __init__(
        self,
        posterior: ConjugatePosterior,
        num_fourier_features: int = 100,
        jitter: float = 1e-6,
    ):
        """Initialize the posterior pathwise sampler.
        
        Args:
            posterior: The conjugate Gaussian process posterior.
            num_fourier_features: Number of Fourier basis functions.
            jitter: Small value for numerical stability.
        """
        # Extract components from posterior
        prior = posterior.prior
        likelihood = posterior.likelihood
        
        # Validate likelihood type
        if not isinstance(likelihood, Gaussian):
            raise TypeError("PosteriorPathwiseSampler requires Gaussian likelihood")
        
        super().__init__(
            prior=prior,
            feature_generator=None,  # Set during sampling
            weight_sampler=None,     # Set during sampling
            jitter=jitter,
        )
        
        self.posterior = posterior
        self.likelihood = likelihood
        self.num_fourier_features = num_fourier_features

    @jaxtyped(typechecker=beartype)
    def sample(
        self,
        num_samples: int,
        key: KeyArray,
        train_data: Dataset,
        **kwargs,
    ) -> FunctionalSample:
        """Generate approximate samples from the GP posterior.
        
        This method replicates the logic from ConjugatePosterior.sample_approx()
        but uses the modular framework for extensibility.
        
        Args:
            num_samples: Number of function samples to generate.
            key: Random key for sampling.
            train_data: Training data for conditioning.
            **kwargs: Additional arguments (unused).
            
        Returns:
            A function that evaluates posterior samples at test inputs.
        """
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
            
        # CRITICAL FIX: Use the same key for both feature generation and weight sampling
        # to exactly match the original implementation in ConjugatePosterior.sample_approx()
        
        # Create composite feature generator (Fourier + canonical)
        # Original uses key for RFF kernel creation at line 576
        self.feature_generator = create_composite_generator(
            self.prior, train_data, self.num_fourier_features, key
        )
        
        # Create posterior weight sampler
        self.weight_sampler = create_posterior_sampler(
            self.prior, self.likelihood, train_data, self.feature_generator
        )
        
        # Sample weights (includes both Fourier and canonical)
        # Original uses the same key for weight sampling at line 578
        weights = self.weight_sampler.sample_weights(
            num_samples, self.feature_generator.num_features, key
        )

        def sample_fn(test_inputs: Float[Array, "N D"]) -> Float[Array, "N B"]:
            """Evaluate posterior samples at test inputs.
            
            Args:
                test_inputs: Input locations for evaluation.
                
            Returns:
                Sample evaluations with shape (N, B).
            """
            # Generate composite features at test inputs
            # Use the same key as for feature generator creation
            features = self.feature_generator.generate_features(test_inputs, key)
            
            # Compute weighted feature sum  
            weighted_features = jnp.inner(features, weights)  # (N, B)
            
            # Add mean function
            mean_values = self.prior.mean_function(test_inputs)  # (N, 1) 
            
            return mean_values + weighted_features
        
        return sample_fn


class FlexiblePathwiseSampler(AbstractPathwiseSampler):
    """Flexible pathwise sampler with configurable components.
    
    Provides maximum flexibility by allowing custom feature generators
    and weight samplers. This enables research and experimentation with
    different pathwise sampling strategies.
    """

    def __init__(
        self,
        prior: AbstractPrior,
        feature_generator: tp.Optional[tp.Any] = None,  # Will be set during sample()
        weight_sampler: tp.Optional[tp.Any] = None,     # Will be set during sample()
        jitter: float = 1e-6,
    ):
        """Initialize the flexible pathwise sampler.
        
        Args:
            prior: The Gaussian process prior.
            feature_generator: Custom feature generator (set during sampling if None).
            weight_sampler: Custom weight sampler (set during sampling if None).
            jitter: Small value for numerical stability.
        """
        super().__init__(
            prior=prior,
            feature_generator=feature_generator,
            weight_sampler=weight_sampler,
            jitter=jitter,
        )

    @jaxtyped(typechecker=beartype)
    def sample(
        self,
        num_samples: int,
        key: KeyArray,
        feature_generator: tp.Optional[tp.Any] = None,
        weight_sampler: tp.Optional[tp.Any] = None,
        **kwargs,
    ) -> FunctionalSample:
        """Generate samples with custom components.
        
        Args:
            num_samples: Number of function samples to generate.
            key: Random key for sampling.
            feature_generator: Custom feature generator (overrides initialization).
            weight_sampler: Custom weight sampler (overrides initialization).
            **kwargs: Additional arguments passed to components.
            
        Returns:
            A function that evaluates samples at test inputs.
        """
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
            
        # Use provided generators or defaults
        if feature_generator is not None:
            self.feature_generator = feature_generator
        if weight_sampler is not None:
            self.weight_sampler = weight_sampler
            
        # Validate that we have required components
        if self.feature_generator is None:
            raise ValueError("feature_generator must be provided")
        if self.weight_sampler is None:
            raise ValueError("weight_sampler must be provided")
        
        # Split keys for different operations
        feature_key, weight_key = jr.split(key)
        
        # Sample weights
        weights = self.weight_sampler.sample_weights(
            num_samples, self.feature_generator.num_features, weight_key
        )

        def sample_fn(test_inputs: Float[Array, "N D"]) -> Float[Array, "N B"]:
            """Evaluate samples at test inputs.
            
            Args:
                test_inputs: Input locations for evaluation.
                
            Returns:
                Sample evaluations with shape (N, B).
            """
            # Generate features
            features = self.feature_generator.generate_features(test_inputs, feature_key)
            
            # Compute weighted features
            weighted_features = jnp.inner(features, weights)
            
            # Add mean function
            mean_values = self.prior.mean_function(test_inputs)
            
            return mean_values + weighted_features
        
        return sample_fn


__all__ = [
    "PriorPathwiseSampler",
    "PosteriorPathwiseSampler", 
    "FlexiblePathwiseSampler",
]