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
"""Utility functions and factory methods for pathwise sampling.

This module provides convenient factory functions, configuration validation,
and utility functions that simplify the creation and use of pathwise samplers.
It serves as the main entry point for most users of the pathwise sampling
functionality.
"""

from __future__ import annotations

import beartype.typing as tp
import warnings
from beartype import beartype
from jaxtyping import jaxtyped

from gpjax.dataset import Dataset
from gpjax.gps import (
    AbstractPrior,
    ConjugatePosterior,
    Prior,
)
from gpjax.likelihoods import Gaussian
from gpjax.sampling.pathwise.base import (
    AbstractPathwiseSampler,
    PathwiseSamplerConfig,
)
from gpjax.sampling.pathwise.samplers import (
    PriorPathwiseSampler,
    PosteriorPathwiseSampler,
    FlexiblePathwiseSampler,
)
from gpjax.typing import KeyArray


class PathwiseSampler:
    """Main factory class for creating pathwise samplers.
    
    This class provides the primary interface for creating pathwise samplers
    with sensible defaults and convenient factory methods. It follows the
    pattern established by other GPJax factory classes.
    """

    @staticmethod
    @jaxtyped(typechecker=beartype)
    def for_prior(
        prior: AbstractPrior,
        num_features: int = 100,
        jitter: float = 1e-6,
        **config,
    ) -> PriorPathwiseSampler:
        """Create a pathwise sampler for GP prior sampling.
        
        This method provides a convenient interface for creating prior samplers
        that replicate Prior.sample_approx() functionality with additional
        configurability.
        
        Example:
        ```python
        import gpjax as gpx
        import jax.random as jr
        
        prior = gpx.gps.Prior(
            mean_function=gpx.mean_functions.Zero(),
            kernel=gpx.kernels.RBF()
        )
        
        sampler = PathwiseSampler.for_prior(prior, num_features=200)
        sample_fn = sampler.sample(num_samples=5, key=jr.PRNGKey(42))
        ```
        
        Args:
            prior: The Gaussian process prior.
            num_features: Number of Fourier basis functions (default: 100).
            jitter: Numerical stability parameter (default: 1e-6).
            **config: Additional configuration options.
            
        Returns:
            Configured prior pathwise sampler.
            
        Raises:
            ValueError: If configuration parameters are invalid.
        """
        # Validate configuration
        _validate_prior_config(num_features, jitter, config)
        
        return PriorPathwiseSampler(
            prior=prior,
            num_features=num_features,
            jitter=jitter,
        )

    @staticmethod
    @jaxtyped(typechecker=beartype) 
    def for_posterior(
        posterior: ConjugatePosterior,
        num_fourier_features: int = 100,
        jitter: float = 1e-6,
        **config,
    ) -> PosteriorPathwiseSampler:
        """Create a pathwise sampler for GP posterior sampling.
        
        This method provides a convenient interface for creating posterior samplers
        that replicate ConjugatePosterior.sample_approx() functionality with
        additional configurability.
        
        Example:
        ```python
        import gpjax as gpx
        import jax.random as jr
        import jax.numpy as jnp
        
        # Create posterior
        prior = gpx.gps.Prior(
            mean_function=gpx.mean_functions.Zero(),
            kernel=gpx.kernels.RBF()
        )
        likelihood = gpx.likelihoods.Gaussian(num_datapoints=50)
        posterior = prior * likelihood
        
        # Create sampler
        sampler = PathwiseSampler.for_posterior(posterior, num_fourier_features=200)
        
        # Generate samples (requires training data)
        train_data = gpx.Dataset(X=X_train, y=y_train)
        sample_fn = sampler.sample(
            num_samples=5, 
            key=jr.PRNGKey(42),
            train_data=train_data
        )
        ```
        
        Args:
            posterior: The conjugate Gaussian process posterior.
            num_fourier_features: Number of Fourier basis functions (default: 100).
            jitter: Numerical stability parameter (default: 1e-6).
            **config: Additional configuration options.
            
        Returns:
            Configured posterior pathwise sampler.
            
        Raises:
            TypeError: If posterior is not a ConjugatePosterior with Gaussian likelihood.
            ValueError: If configuration parameters are invalid.
        """
        # Validate posterior type
        if not isinstance(posterior, ConjugatePosterior):
            raise TypeError("Posterior sampling requires a ConjugatePosterior")
        
        if not isinstance(posterior.likelihood, Gaussian):
            raise TypeError("Posterior sampling requires a Gaussian likelihood")
        
        # Validate configuration
        _validate_posterior_config(num_fourier_features, jitter, config)
        
        return PosteriorPathwiseSampler(
            posterior=posterior,
            num_fourier_features=num_fourier_features,
            jitter=jitter,
        )

    @staticmethod
    @jaxtyped(typechecker=beartype)
    def flexible(
        prior: AbstractPrior,
        jitter: float = 1e-6,
        **config,
    ) -> FlexiblePathwiseSampler:
        """Create a flexible pathwise sampler for custom configurations.
        
        This method creates a sampler that accepts custom feature generators
        and weight samplers at sampling time, enabling research and experimentation.
        
        Args:
            prior: The Gaussian process prior.
            jitter: Numerical stability parameter (default: 1e-6).
            **config: Additional configuration options.
            
        Returns:
            Configured flexible pathwise sampler.
        """
        return FlexiblePathwiseSampler(
            prior=prior,
            jitter=jitter,
        )


def validate_config(config: PathwiseSamplerConfig) -> None:
    """Validate a pathwise sampler configuration.
    
    Args:
        config: Configuration dictionary to validate.
        
    Raises:
        ValueError: If any configuration values are invalid.
    """
    if "num_fourier_features" in config:
        if config["num_fourier_features"] <= 0:
            raise ValueError("num_fourier_features must be positive")
    
    if "num_canonical_features" in config:
        if config["num_canonical_features"] < 0:
            raise ValueError("num_canonical_features must be non-negative")
    
    if "jitter" in config:
        if config["jitter"] <= 0:
            raise ValueError("jitter must be positive")
    
    if "weight_strategy" in config:
        valid_strategies = ["gaussian", "posterior"]
        if config["weight_strategy"] not in valid_strategies:
            raise ValueError(f"weight_strategy must be one of {valid_strategies}")
    
    if "feature_strategy" in config:
        valid_strategies = ["fourier", "canonical", "composite"]
        if config["feature_strategy"] not in valid_strategies:
            raise ValueError(f"feature_strategy must be one of {valid_strategies}")


def _validate_prior_config(
    num_features: int,
    jitter: float,
    config: tp.Dict[str, tp.Any],
) -> None:
    """Validate configuration for prior sampling."""
    if num_features <= 0:
        raise ValueError("num_features must be positive")
    
    if jitter <= 0:
        raise ValueError("jitter must be positive")
    
    # Check for common mistakes
    if "num_fourier_features" in config:
        warnings.warn(
            "num_fourier_features in config ignored for prior sampling. "
            "Use num_features parameter instead.",
            UserWarning,
        )
    
    if "train_data" in config:
        warnings.warn(
            "train_data in config ignored for prior sampling.",
            UserWarning,
        )


def _validate_posterior_config(
    num_fourier_features: int,
    jitter: float,
    config: tp.Dict[str, tp.Any],
) -> None:
    """Validate configuration for posterior sampling.""" 
    if num_fourier_features <= 0:
        raise ValueError("num_fourier_features must be positive")
    
    if jitter <= 0:
        raise ValueError("jitter must be positive")


def create_default_config(
    sampler_type: str = "prior",
    **overrides,
) -> PathwiseSamplerConfig:
    """Create a default configuration for pathwise sampling.
    
    Args:
        sampler_type: Type of sampler ("prior" or "posterior").
        **overrides: Configuration values to override defaults.
        
    Returns:
        Default configuration with any overrides applied.
    """
    if sampler_type == "prior":
        config = PathwiseSamplerConfig(
            num_fourier_features=100,
            jitter=1e-6,
            weight_strategy="gaussian",
            feature_strategy="fourier",
        )
    elif sampler_type == "posterior":
        config = PathwiseSamplerConfig(
            num_fourier_features=100,
            num_canonical_features=0,  # Determined by training data size
            jitter=1e-6,
            weight_strategy="posterior", 
            feature_strategy="composite",
        )
    else:
        raise ValueError(f"Unknown sampler_type: {sampler_type}")
    
    # Apply overrides
    config.update(overrides)
    
    return config


def estimate_feature_count(
    train_size: int,
    total_budget: int = 200,
    fourier_ratio: float = 0.7,
) -> tp.Tuple[int, int]:
    """Estimate good feature counts for posterior sampling.
    
    Provides heuristics for allocating features between Fourier and canonical
    components based on training set size and computational budget.
    
    Args:
        train_size: Number of training points.
        total_budget: Total computational budget for features.
        fourier_ratio: Fraction of budget to allocate to Fourier features.
        
    Returns:
        Tuple of (num_fourier_features, num_canonical_features).
    """
    num_fourier = int(total_budget * fourier_ratio)
    num_canonical = min(train_size, total_budget - num_fourier)
    
    # Ensure we don't exceed the budget
    if num_fourier + num_canonical > total_budget:
        num_canonical = total_budget - num_fourier
    
    return num_fourier, num_canonical


def backward_compatibility_wrapper(
    gp_or_posterior: tp.Union[Prior, ConjugatePosterior],
    sample_approx_method: tp.Callable,
) -> tp.Callable:
    """Wrap existing sample_approx methods for backward compatibility.
    
    This function enables gradual migration from existing sample_approx methods
    to the new pathwise sampling framework while maintaining API compatibility.
    
    Args:
        gp_or_posterior: The GP prior or posterior object.
        sample_approx_method: The original sample_approx method.
        
    Returns:
        Wrapped method that can optionally use pathwise sampling.
    """
    def wrapped_sample_approx(*args, use_pathwise: bool = False, **kwargs):
        """Wrapped sample_approx with optional pathwise sampling."""
        if use_pathwise:
            # Use new pathwise implementation
            if isinstance(gp_or_posterior, Prior):
                sampler = PathwiseSampler.for_prior(gp_or_posterior)
                return sampler.sample(*args, **kwargs)
            elif isinstance(gp_or_posterior, ConjugatePosterior):
                sampler = PathwiseSampler.for_posterior(gp_or_posterior)
                return sampler.sample(*args, **kwargs)
            else:
                raise TypeError("Unsupported GP type for pathwise sampling")
        else:
            # Use original implementation
            return sample_approx_method(*args, **kwargs)
    
    return wrapped_sample_approx


__all__ = [
    "PathwiseSampler",
    "validate_config",
    "create_default_config",
    "estimate_feature_count", 
    "backward_compatibility_wrapper",
]