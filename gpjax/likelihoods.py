from objax import Module
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
from .parameters import Parameter
from typing import Optional, Tuple
from tensorflow_probability.substrates.jax import distributions as tfd


class Likelihood(Module):
    """
    Base class for all likelihood functions. By inheriting the `Module` class from Objax, seamless interaction with
    model parameters is provided.
    """
    def __init__(self, name: Optional[str] = "Likelihood"):
        """
        Args:
            name: Optional naming of the likelihood.
        """
        self.name = name

    def __call__(self):
        raise NotImplementedError

    def log_density(self, F, Y) -> jnp.ndarray:
        raise NotImplementedError

    def predictive_moments(self, latent_mean,
                           latent_variance) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute the predictive mean and predictive variance using the mean and variance of GP's latent function.

        Args:
            mean: Mean of the latent function
            variance: Variance of the latent function

        Returns:
            Tuple containing the predictive mean and variance of the process.
        """
        raise NotImplementedError


class Gaussian(Likelihood):
    """
    Gaussian likelihood function 
    """
    def __init__(self, noise: jnp.array = jnp.array([1.0]), name="Gaussian"):
        super().__init__(name=name)
        self.noise = Parameter(noise)

    def log_likelihood(self, x: jnp.ndarray, mu: jnp.ndarray, L: jnp.ndarray):
        delta = x - mu
        alpha = solve_triangular(L, delta, lower=True)
        L_diag = jnp.sum(jnp.log(jnp.diag(L)))
        ll = -0.5 * jnp.sum(jnp.square(alpha), axis=0)
        ll -= 0.5 * jnp.log(2 * jnp.pi)
        ll -= L_diag
        return ll

    def __call__(self):
        raise NotImplementedError


class Bernoulli(Likelihood):
    def __init__(self):
        super().__init__(name="Bernoulli")
        self.random_variable = tfd.ProbitBernoulli

    def log_density(self, F, Y) -> jnp.ndarray:
        return self.random_variable(F).log_prob(Y)

    def predictive_moments(self, latent_mean,
                           latent_variance) -> Tuple[jnp.ndarray, jnp.ndarray]:
        r"""
        Using the predictive mean and variance of the latent function :math:`f^{\star}`, we can analytically compute
        the averaged predictive probability (eq. 3.25 [Rasmussen and Williams (2006)]) as we've used the Probit link
        function. For a full derivation, see Appendix 3.9 of Rasmussen and Williams (2006). However, the first moment
        is given by
        .. math::
            \mathbb{E}_q[x] = \mu + \frac{\sigma^2 \mathcal{N}(z)}{\Phi(z)\nu\sqrt{1 + \frac{\sigma^2}{\nu^2}}}

        and the second moment by
        .. math::
            \mathbb{V}[x] = \mathbb{E}_q[x^2] - [\mathbb{E}_q[x]]^2.

        Args:
            mean: Mean of the latent function
            variance: Variance of the latent function

        Returns:
            Tuple containing the predictive mean and variance of the process.
        """
        rv = self.random_variable(latent_mean.ravel() /
                                  jnp.sqrt(1 + latent_variance.ravel()))
        return rv.mean(), rv.variance()
