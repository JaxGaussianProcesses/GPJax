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
"""Feature generators for pathwise sampling.

This module implements feature generators that produce basis functions for
finite feature approximations of Gaussian processes. The generators follow
the methodology of Wilson et al. (2020) and extract logic from existing
GPJax sampling implementations.

Two main types of features are supported:
1. Fourier Features: Random Fourier Features (RFF) from kernel spectral densities
2. Canonical Features: Kernel evaluations at training points for posterior sampling

References:
    Wilson, J., Borovitskiy, V., Terenin, A., Mostowsky, P., & Deisenroth, M. (2020).
    Efficiently sampling functions from Gaussian process posteriors.
    International Conference on Machine Learning.
"""

from __future__ import annotations

import beartype.typing as tp
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import (
    Array,
    Float,
    jaxtyped,
)

from gpjax.dataset import Dataset
from gpjax.gps import AbstractPrior
from gpjax.kernels import RFF
from gpjax.sampling.pathwise.base import AbstractFeatureGenerator
from gpjax.typing import KeyArray


class FourierFeatureGenerator(AbstractFeatureGenerator):
    """Generator for Random Fourier Features (RFF).
    
    Generates features by sampling from the spectral density of a stationary
    kernel and constructing the corresponding Fourier features. This follows
    the approach of Rahimi & Recht (2008) and replicates the logic from
    Prior.sample_approx().
    
    The features take the form:
        φ(x) = √(2/M) * [cos(ω₁ᵀx + b₁), ..., cos(ωₘᵀx + bₘ), sin(ω₁ᵀx + b₁), ..., sin(ωₘᵀx + bₘ)]
    
    where ωᵢ are sampled from the kernel's spectral density and bᵢ ~ Uniform[0, 2π].
    """

    def __init__(
        self,
        prior: AbstractPrior,
        num_features: int,
        key: KeyArray,
    ):
        """Initialize the Fourier feature generator.
        
        Args:
            prior: The Gaussian process prior containing the kernel.
            num_features: Number of Fourier basis functions to generate.
            key: Random key for sampling frequencies and phases.
        """
        if num_features <= 0:
            raise ValueError("num_features must be positive")
            
        self.prior = prior
        self._num_features = num_features
        self.key = key
        
        # Store parameters for lazy RFF creation
        self._rff_kernels = {}  # Cache by dimension
        self._base_kernel = prior.kernel

    @property
    def num_features(self) -> int:
        """Number of Fourier features (2 * num_basis_fns for cos + sin)."""
        return 2 * self._num_features

    def _get_rff_kernel(self, test_inputs: Float[Array, "N D"]) -> RFF:
        """Get or create RFF kernel for the given input dimensions."""
        # Check if we're inside a JAX transformation by checking for tracers
        import jax
        if any(isinstance(x, jax.core.Tracer) for x in jax.tree_util.tree_leaves(test_inputs)):
            # We're inside JIT/vmap/etc - don't create new kernels
            n_dims = test_inputs.shape[1]  # This will be a traced value
            if n_dims not in self._rff_kernels:
                raise RuntimeError(
                    f"RFF kernel for {n_dims}-dimensional inputs not pre-initialized. "
                    f"When using JIT, call precompute_for_dimensions({n_dims}) first, "
                    f"or evaluate the function once without JIT to initialize."
                )
            return self._rff_kernels[n_dims]
        
        # Not inside a transform - safe to create kernel
        n_dims = int(test_inputs.shape[1])
        
        if n_dims not in self._rff_kernels:
            # Create kernel with proper dimensions
            import copy
            base_kernel = copy.deepcopy(self._base_kernel)
            base_kernel.n_dims = n_dims
            
            self._rff_kernels[n_dims] = RFF(
                base_kernel=base_kernel,
                num_basis_fns=self._num_features,
                key=self.key,
            )
        
        return self._rff_kernels[n_dims]

    def precompute_for_dimensions(self, n_dims: int) -> None:
        """Pre-compute RFF kernel for given input dimensions.
        
        This should be called before using the generator in JAX transformations
        to avoid tracer issues.
        
        Args:
            n_dims: Number of input dimensions to prepare for.
        """
        if n_dims not in self._rff_kernels:
            import copy
            base_kernel = copy.deepcopy(self._base_kernel)
            base_kernel.n_dims = n_dims
            
            self._rff_kernels[n_dims] = RFF(
                base_kernel=base_kernel,
                num_basis_fns=self._num_features,
                key=self.key,
            )

    @jaxtyped(typechecker=beartype)
    def generate_features(
        self,
        test_inputs: Float[Array, "N D"],
        key: KeyArray,
    ) -> Float[Array, "N F"]:
        """Generate Fourier feature evaluations.
        
        Extracts and replicates the logic from _build_fourier_features_fn
        in the existing GPJax codebase.
        
        Args:
            test_inputs: Input locations for feature evaluation.
            key: Random key (not used - frequencies fixed at initialization).
            
        Returns:
            Fourier features with shape (N, 2*M) where M is num_basis_fns.
        """
        # Get RFF kernel for the correct dimensions
        rff_kernel = self._get_rff_kernel(test_inputs)
        
        # Compute features using RFF kernel
        features = rff_kernel.compute_features(test_inputs)
        
        # Scale features by kernel variance (matching existing implementation)
        variance_scale = jnp.sqrt(self.prior.kernel.variance.value / self._num_features)
        features = features * variance_scale
        
        return features


class CanonicalFeatureGenerator(AbstractFeatureGenerator):
    """Generator for canonical features from training data.
    
    Generates features by evaluating the kernel at training input locations.
    These features are essential for posterior sampling as they capture the
    transition from prior to posterior. This extracts logic from
    ConjugatePosterior.sample_approx().
    
    The canonical features are:
        ψⱼ(x) = k(x, xⱼ)
    
    where xⱼ are the training inputs and k is the kernel function.
    """

    def __init__(
        self,
        prior: AbstractPrior,
        train_data: Dataset,
    ):
        """Initialize the canonical feature generator.
        
        Args:
            prior: The Gaussian process prior containing the kernel.
            train_data: Training dataset providing canonical feature locations.
        """
        self.prior = prior
        self.train_data = train_data
        self._num_features = train_data.n

    @property
    def num_features(self) -> int:
        """Number of canonical features (equals number of training points)."""
        return self._num_features

    @jaxtyped(typechecker=beartype)
    def generate_features(
        self,
        test_inputs: Float[Array, "N D"],
        key: KeyArray,
    ) -> Float[Array, "N F"]:
        """Generate canonical feature evaluations.
        
        Computes kernel cross-covariance between test inputs and training inputs.
        This replicates the canonical feature computation from existing posterior
        sampling implementation.
        
        Args:
            test_inputs: Input locations for feature evaluation.
            key: Random key (not used - features are deterministic).
            
        Returns:
            Canonical features with shape (N, F) where F is number of training points.
        """
        # Compute cross-covariance k(test_inputs, train_inputs)
        canonical_features = self.prior.kernel.cross_covariance(
            test_inputs, self.train_data.X
        )
        
        return canonical_features


class CompositeFeatureGenerator(AbstractFeatureGenerator):
    """Composite generator combining multiple feature types.
    
    Combines Fourier and canonical features for posterior sampling, following
    the approach of Wilson et al. (2020). This enables accurate approximation
    of GP posteriors by combining global (Fourier) and local (canonical) features.
    
    The combined features are:
        φ(x) = [φ_fourier(x), φ_canonical(x)]
    """

    def __init__(
        self,
        fourier_generator: FourierFeatureGenerator,
        canonical_generator: CanonicalFeatureGenerator,
    ):
        """Initialize the composite feature generator.
        
        Args:
            fourier_generator: Generator for Fourier features.
            canonical_generator: Generator for canonical features.
        """
        self.fourier_generator = fourier_generator
        self.canonical_generator = canonical_generator

    @property
    def num_features(self) -> int:
        """Total number of features (Fourier + canonical)."""
        return (
            self.fourier_generator.num_features
            + self.canonical_generator.num_features
        )

    @jaxtyped(typechecker=beartype)
    def generate_features(
        self,
        test_inputs: Float[Array, "N D"],
        key: KeyArray,
    ) -> Float[Array, "N F"]:
        """Generate combined feature evaluations.
        
        Args:
            test_inputs: Input locations for feature evaluation.
            key: Random key for feature generation.
            
        Returns:
            Combined features with shape (N, F_fourier + F_canonical).
        """
        # Generate Fourier features
        fourier_features = self.fourier_generator.generate_features(test_inputs, key)
        
        # Generate canonical features  
        canonical_features = self.canonical_generator.generate_features(test_inputs, key)
        
        # Concatenate features
        combined_features = jnp.concatenate([fourier_features, canonical_features], axis=-1)
        
        return combined_features


def create_fourier_generator(
    prior: AbstractPrior,
    num_features: int,
    key: KeyArray,
) -> FourierFeatureGenerator:
    """Factory function for creating Fourier feature generators.
    
    Args:
        prior: The Gaussian process prior.
        num_features: Number of Fourier basis functions.
        key: Random key for feature generation.
        
    Returns:
        Configured Fourier feature generator.
    """
    return FourierFeatureGenerator(prior=prior, num_features=num_features, key=key)


def create_canonical_generator(
    prior: AbstractPrior,
    train_data: Dataset,
) -> CanonicalFeatureGenerator:
    """Factory function for creating canonical feature generators.
    
    Args:
        prior: The Gaussian process prior.
        train_data: Training dataset for canonical features.
        
    Returns:
        Configured canonical feature generator.
    """
    return CanonicalFeatureGenerator(prior=prior, train_data=train_data)


def create_composite_generator(
    prior: AbstractPrior,
    train_data: Dataset,
    num_fourier_features: int,
    key: KeyArray,
) -> CompositeFeatureGenerator:
    """Factory function for creating composite feature generators.
    
    Args:
        prior: The Gaussian process prior.
        train_data: Training dataset for canonical features.
        num_fourier_features: Number of Fourier basis functions.
        key: Random key for Fourier feature generation.
        
    Returns:
        Configured composite feature generator.
    """
    fourier_gen = create_fourier_generator(prior, num_fourier_features, key)
    canonical_gen = create_canonical_generator(prior, train_data)
    return CompositeFeatureGenerator(fourier_gen, canonical_gen)


__all__ = [
    "FourierFeatureGenerator",
    "CanonicalFeatureGenerator", 
    "CompositeFeatureGenerator",
    "create_fourier_generator",
    "create_canonical_generator",
    "create_composite_generator",
]