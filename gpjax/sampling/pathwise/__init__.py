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
"""Pathwise sampling for Gaussian processes.

This module implements pathwise sampling following Wilson et al. (2020),
providing efficient functional sampling from GP priors and posteriors using
finite feature approximations.

The main entry point is the PathwiseSampler factory class:

```python
import gpjax as gpx
import jax.random as jr

# Prior sampling
prior = gpx.gps.Prior(mean_function=gpx.mean_functions.Zero(), kernel=gpx.kernels.RBF())
sampler = gpx.sampling.PathwiseSampler.for_prior(prior)
sample_fn = sampler.sample(num_samples=5, key=jr.PRNGKey(42))

# Posterior sampling  
posterior = prior * gpx.likelihoods.Gaussian(num_datapoints=50)
sampler = gpx.sampling.PathwiseSampler.for_posterior(posterior)
sample_fn = sampler.sample(num_samples=5, key=jr.PRNGKey(42), train_data=data)
```

References:
    Wilson, J., Borovitskiy, V., Terenin, A., Mostowsky, P., & Deisenroth, M. (2020).
    Efficiently sampling functions from Gaussian process posteriors.
    International Conference on Machine Learning.
"""

from gpjax.sampling.pathwise.base import (
    AbstractFeatureGenerator,
    AbstractPathwiseSampler,
    AbstractWeightSampler,
    PathwiseSamplerConfig,
)
from gpjax.sampling.pathwise.features import (
    CanonicalFeatureGenerator,
    CompositeFeatureGenerator,
    FourierFeatureGenerator,
    create_canonical_generator,
    create_composite_generator,
    create_fourier_generator,
)
from gpjax.sampling.pathwise.samplers import (
    FlexiblePathwiseSampler,
    PosteriorPathwiseSampler,
    PriorPathwiseSampler,
)
from gpjax.sampling.pathwise.utils import (
    PathwiseSampler,
    create_default_config,
    estimate_feature_count,
    validate_config,
)
from gpjax.sampling.pathwise.weights import (
    GaussianWeightSampler,
    PosteriorWeightSampler,
    create_gaussian_sampler,
    create_posterior_sampler,
)

__all__ = [
    # Main interface
    "PathwiseSampler",
    
    # Concrete samplers
    "PriorPathwiseSampler",
    "PosteriorPathwiseSampler", 
    "FlexiblePathwiseSampler",
    
    # Base abstractions
    "AbstractPathwiseSampler",
    "AbstractFeatureGenerator",
    "AbstractWeightSampler",
    
    # Feature generators
    "FourierFeatureGenerator",
    "CanonicalFeatureGenerator",
    "CompositeFeatureGenerator",
    
    # Weight samplers  
    "GaussianWeightSampler",
    "PosteriorWeightSampler",
    
    # Factory functions
    "create_fourier_generator",
    "create_canonical_generator", 
    "create_composite_generator",
    "create_gaussian_sampler",
    "create_posterior_sampler",
    
    # Utilities
    "PathwiseSamplerConfig",
    "validate_config",
    "create_default_config",
    "estimate_feature_count",
]