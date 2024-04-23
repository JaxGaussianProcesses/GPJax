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

import abc
from dataclasses import dataclass

from beartype.typing import (
    Any,
    Union,
)
from jax import vmap
from jax import custom_jvp
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Float
import tensorflow_probability.substrates.jax as tfp
import jax
from gpjax.base import (
    Module,
    param_field,
    static_field,
)
from gpjax.distributions import GaussianDistribution
from gpjax.integrators import (
    AbstractIntegrator,
    AnalyticalGaussianIntegrator,
    GHQuadratureIntegrator,
    TwoDimGHQuadratureIntegrator,
    ThreeDimGHQuadratureIntegrator,
)
from gpjax.typing import (
    Array,
    ScalarFloat,
)

tfb = tfp.bijectors
tfd = tfp.distributions


@dataclass
class AbstractLikelihood(Module):
    r"""Abstract base class for likelihoods."""

    num_datapoints: int = static_field()
    integrator: AbstractIntegrator = static_field(GHQuadratureIntegrator())

    def __call__(self, *args: Any, **kwargs: Any) -> tfd.Distribution:
        r"""Evaluate the likelihood function at a given predictive distribution.

        Args:
            *args (Any): Arguments to be passed to the likelihood's `predict` method.
            **kwargs (Any): Keyword arguments to be passed to the likelihood's
                `predict` method.

        Returns
        -------
            tfd.Distribution: The predictive distribution.
        """
        return self.predict(*args, **kwargs)

    @abc.abstractmethod
    def predict(self, *args: Any, **kwargs: Any) -> tfd.Distribution:
        r"""Evaluate the likelihood function at a given predictive distribution.

        Args:
            *args (Any): Arguments to be passed to the likelihood's `predict` method.
            **kwargs (Any): Keyword arguments to be passed to the likelihood's
                `predict` method.

        Returns
        -------
            tfd.Distribution: The predictive distribution.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def link_function(self, f: Float[Array, "..."]) -> tfd.Distribution:
        r"""Return the link function of the likelihood function.

        Returns
        -------
            tfd.Distribution: The distribution of observations, y, given values of the
                Gaussian process, f.
        """
        raise NotImplementedError

    def expected_log_likelihood(
        self,
        y: Float[Array, "N n"],
        mean: Float[Array, "N L n"],
        variance: Float[Array, "N L n"],
    ) -> Float[Array, " N"]:
        r"""Compute the expected log likelihood.

        For a variational distribution $`q(f)\sim\mathcal{N}(m, s)`$ and a likelihood
        $`p(y|f)`$, compute the expected log likelihood:
        ```math
        \mathbb{E}_{q(f)}\left[\log p(y|f)\right]
        ```

        Args:
            y (Float[Array, 'N D']): The observed response variable.
            mean (Float[Array, 'N D']): The variational mean.
            variance (Float[Array, 'N D']): The variational variance.

        Returns:
            ScalarFloat: The expected log likelihood.
        """
        log_prob = vmap(lambda f, y: self.link_function(f).log_prob(y))
        return self.integrator(
            fun=log_prob, y=y, mean=mean, variance=variance, likelihood=self
        )


@dataclass
class Gaussian(AbstractLikelihood):
    r"""Gaussian likelihood object.

    Args:
        obs_stddev (Union[ScalarFloat, Float[Array, "#N"]]): the standard deviation
            of the Gaussian observation noise.

    """

    obs_stddev: Union[ScalarFloat, Float[Array, "#N"]] = param_field(
        jnp.array(1.0), bijector=tfb.Softplus()
    )
    integrator: AbstractIntegrator = static_field(AnalyticalGaussianIntegrator())

    def link_function(self, f: Float[Array, "..."]) -> tfd.Normal:
        r"""The link function of the Gaussian likelihood.

        Args:
            f (Float[Array, "..."]): Function values.

        Returns
        -------
            tfd.Normal: The likelihood function.
        """
        return tfd.Normal(loc=f, scale=self.obs_stddev.astype(f.dtype))

    def predict(
        self, dist: Union[tfd.MultivariateNormalTriL, GaussianDistribution]
    ) -> tfd.MultivariateNormalFullCovariance:
        r"""Evaluate the Gaussian likelihood.

        Evaluate the Gaussian likelihood function at a given predictive
        distribution. Computationally, this is equivalent to summing the
        observation noise term to the diagonal elements of the predictive
        distribution's covariance matrix.

        Args:
            dist (tfd.Distribution): The Gaussian process posterior,
                evaluated at a finite set of test points.

        Returns
        -------
            tfd.Distribution: The predictive distribution.
        """
        n_data = dist.event_shape[0]
        cov = dist.covariance()
        noisy_cov = cov.at[jnp.diag_indices(n_data)].add(self.obs_stddev**2)

        return tfd.MultivariateNormalFullCovariance(dist.mean(), noisy_cov)


@dataclass
class Bernoulli(AbstractLikelihood):
    def link_function(self, f: Float[Array, "..."]) -> tfd.Distribution:
        r"""The probit link function of the Bernoulli likelihood.

        Args:
            f (Float[Array, "..."]): Function values.

        Returns
        -------
            tfd.Distribution: The likelihood function.
        """
        return tfd.Bernoulli(probs=inv_probit(f))

    def predict(self, dist: tfd.Distribution) -> tfd.Distribution:
        r"""Evaluate the pointwise predictive distribution.

        Evaluate the pointwise predictive distribution, given a Gaussian
        process posterior and likelihood parameters.

        Args:
            dist (tfd.Distribution): The Gaussian process posterior, evaluated
                at a finite set of test points.

        Returns
        -------
            tfd.Distribution: The pointwise predictive distribution.
        """
        variance = jnp.diag(dist.covariance())
        mean = dist.mean().ravel()
        return self.link_function(mean / jnp.sqrt(1.0 + variance))



@dataclass
class LogitBernoulli(AbstractLikelihood):
    def link_function(self, f: Float[Array, "..."]) -> tfd.Distribution:
        prob = jnp.exp(f) / (1 + jnp.exp(f))
        return tfd.Bernoulli(probs=prob)

    def predict(self, dist: tfd.Distribution) -> tfd.Distribution:
        raise NotImplementedError



@dataclass
class Poisson(AbstractLikelihood):
    def link_function(self, f: Float[Array, "..."]) -> tfd.Distribution:
        r"""The link function of the Poisson likelihood.

        Args:
            f (Float[Array, "..."]): Function values.

        Returns:
            tfd.Distribution: The likelihood function.
        """
        return tfd.Poisson(rate=jnp.exp(f))

    def predict(self, dist: tfd.Distribution) -> tfd.Distribution:
        r"""Evaluate the pointwise predictive distribution.

        Evaluate the pointwise predictive distribution, given a Gaussian
        process posterior and likelihood parameters.

        Args:
            dist (tfd.Distribution): The Gaussian process posterior, evaluated
                at a finite set of test points.

        Returns:
            tfd.Distribution: The pointwise predictive distribution.
        """
        return self.link_function(dist.mean())


def inv_probit(x: Float[Array, " *N"]) -> Float[Array, " *N"]:
    r"""Compute the inverse probit function.

    Args:
        x (Float[Array, "*N"]): A vector of values.

    Returns
    -------
        Float[Array, "*N"]: The inverse probit of the input vector.
    """
    jitter = 1e-3  # To ensure output is in interval (0, 1).
    return 0.5 * (1.0 + jsp.special.erf(x / jnp.sqrt(2.0))) * (1 - 2 * jitter) + jitter





@dataclass
class Exponential(AbstractLikelihood):
    def link_function(self, f: Float[Array, "..."]) -> tfd.Distribution:
        r"""The link function of the Poisson likelihood.

        Args:
            f (Float[Array, "..."]): Function values.

        Returns:
            tfd.Distribution: The likelihood function.
        """
        return tfd.Exponential(rate=jnp.exp(-f))

    def predict(self, dist: tfd.Distribution) -> tfd.Distribution:
        r"""Evaluate the pointwise predictive distribution.

        Evaluate the pointwise predictive distribution, given a Gaussian
        process posterior and likelihood parameters.

        Args:
            dist (tfd.Distribution): The Gaussian process posterior, evaluated
                at a finite set of test points.

        Returns:
            tfd.Distribution: The pointwise predictive distribution.
        """
        raise NotImplementedError



@dataclass
class Gamma(AbstractLikelihood):
    
    scale1: Union[ScalarFloat, Float[Array, "N"]] = param_field(
        jnp.array(1.0), bijector=tfb.Softplus()
    )
    
    def link_function(self, f: Float[Array, "L n"]) -> tfd.Distribution:
        r"""The link function of the Poisson likelihood.

        Args:
            f (Float[Array, "..."]): Function values.

        Returns:
            tfd.Distribution: The likelihood function.
        """
        assert jnp.shape(f)[0]==1
        #return tfd.Gamma(concentration=self.scale1, rate=jnp.exp(-f[0,:]))
        return tfd.Gamma(concentration=jnp.exp(-f[0,:]), rate=self.scale1)

    def predict(self, dist: tfd.Distribution) -> tfd.Distribution:
        r"""Evaluate the pointwise predictive distribution.

        Evaluate the pointwise predictive distribution, given a Gaussian
        process posterior and likelihood parameters.

        Args:
            dist (tfd.Distribution): The Gaussian process posterior, evaluated
                at a finite set of test points.

        Returns:
            tfd.Distribution: The pointwise predictive distribution.
        """
        raise NotImplementedError




@dataclass
class Gamma2(AbstractLikelihood):
    integrator: AbstractIntegrator = static_field(TwoDimGHQuadratureIntegrator())
    initial_scale: Union[ScalarFloat, Float[Array, "N"]] = static_field(jnp.array(1.0))

    def link_function(self, f: Float[Array, "L n"]) -> tfd.Distribution:
        assert jnp.shape(f)[0]==2
        #return tfd.Gamma(concentration=self.initial_scale*jnp.exp(f[1,:]), rate=jnp.exp(-f[0,:]))
        return tfd.Gamma(concentration=jnp.exp(-f[0,:]-f[1,:]), rate=self.initial_scale*jnp.exp(-f[1,:]))

    def predict(self, dist: tfd.Distribution) -> tfd.Distribution:
        raise NotImplementedError








class FiddleMixture(tfd.Mixture):
    

    def log_prob(self, x): # [B, 1]
        log_probs = jnp.log(self.cat.probs)[...,1] # [n]
        log_gamma_probs = self.components[0].log_prob(jnp.clip(x,1e-20)) # zeros will be masked anyqay
    
        return _log_prob(x, log_probs, log_gamma_probs)
        

    
@custom_jvp
def _log_prob(x, a, b):
    return  jnp.where(x==0,a, (1-a)*b)
    
@_log_prob.defjvp
def _log_prop_jvp(primals, tangents):
    x, a, b = primals
    x_dot, a_dot, b_dot = tangents
    primal_out = _log_prob(x, a, b)
    tangent_out =  jnp.where(x==0,a_dot, b_dot*(1-a) -a_dot*b) +x_dot # NOTE THIS DOESNT WORK FOR Xdot
    return primal_out, tangent_out
    




@dataclass
class Bernoulli(AbstractLikelihood):
    def link_function(self, f: Float[Array, "..."]) -> tfd.Distribution:
        r"""The probit link function of the Bernoulli likelihood.

        Args:
            f (Float[Array, "..."]): Function values.

        Returns
        -------
            tfd.Distribution: The likelihood function.
        """
        probs = inv_probit(f)
        #probs = jnp.clip(f,1e-5,1.0-1e-5)
        return tfd.Bernoulli(probs=probs)

    def predict(self, dist: tfd.Distribution) -> tfd.Distribution:
        r"""Evaluate the pointwise predictive distribution.

        Evaluate the pointwise predictive distribution, given a Gaussian
        process posterior and likelihood parameters.

        Args:
            dist (tfd.Distribution): The Gaussian process posterior, evaluated
                at a finite set of test points.

        Returns
        -------
            tfd.Distribution: The pointwise predictive distribution.
        """
        variance = jnp.diag(dist.covariance())
        mean = dist.mean().ravel()
        return self.link_function(mean / jnp.sqrt(1.0 + variance))




        

@dataclass
class BernoulliGamma(AbstractLikelihood):
    integrator: AbstractIntegrator = static_field(TwoDimGHQuadratureIntegrator())
    initial_scale: Union[ScalarFloat, Float[Array, "N"]] = static_field(jnp.array(1.0))
    initial_scale_2: Union[ScalarFloat, Float[Array, "N"]] = static_field(jnp.array(1.0))

    def link_function(self, f: Float[Array, "L n"]) -> tfd.Distribution:
        assert jnp.shape(f)[0]==2
        prob = inv_probit(f[0,:])
        # f_1 =
        # prob = jnp.clip(f[0,:],1e-5,1.0-1e-5)
        # f_1 = jnp.clip(f[1,:],1e-5)

        #gamma =  tfd.Gamma(concentration=self.initial_scale, rate=self.initial_scale_2 * f_1)
        gamma = tfd.Gamma(concentration=jnp.exp(-f[1,:]), rate=self.initial_scale)
        bernoulli_gamma = FiddleMixture(
            cat=tfd.Categorical(probs=jnp.stack([prob, 1.-prob],-1)),
                components=[gamma,tfd.Deterministic(jnp.zeros_like(gamma.mean()))]
                )
        return bernoulli_gamma
    
    def predict(self, dist: tfd.Distribution) -> tfd.Distribution:
        raise NotImplementedError

@dataclass
class BernoulliGamma2(AbstractLikelihood):
    integrator: AbstractIntegrator = static_field(ThreeDimGHQuadratureIntegrator())
    initial_scale: Union[ScalarFloat, Float[Array, "N"]] = static_field(jnp.array(1.0))
    initial_scale_2: Union[ScalarFloat, Float[Array, "N"]] = static_field(jnp.array(1.0))
    

    def link_function(self, f: Float[Array, "L n"]) -> tfd.Distribution:
        assert jnp.shape(f)[0]==3
        prob = inv_probit(f[0,:])
        # f_1 = jnp.exp(-f[1,:])
        # f_2 = jnp.exp(f[2,:])
        
        # prob = jnp.clip(f[0,:],1e-5,1.0-1e-5)
        # f_1 = jnp.clip(f[1,:],1e-5)
        # f_2 = jnp.clip(f[2,:],1e-5)
        
        #gamma =  tfd.Gamma(concentration=self.initial_scale*f_2, rate=self.initial_scale_2 * f_1)
        gamma = tfd.Gamma(concentration=jnp.exp(-f[1,:]-f[2,:]), rate=self.initial_scale*jnp.exp(-f[2,:]))
        bernoulli_gamma = FiddleMixture(
            cat=tfd.Categorical(probs=jnp.stack([prob, 1.-prob],-1)),
                components=[gamma,tfd.Deterministic(jnp.zeros_like(gamma.mean()))]
                )
        return bernoulli_gamma
    
    def predict(self, dist: tfd.Distribution) -> tfd.Distribution:
        raise NotImplementedError






NonGaussian = Union[Poisson, Bernoulli]

__all__ = [
    "AbstractLikelihood",
    "NonGaussian",
    "Gaussian",
    "Bernoulli",
    "Poisson",
    "inv_probit",
]
