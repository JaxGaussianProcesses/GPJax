import jax.numpy as jnp
import jax.random as jr
from objax import Module
from typing import Optional, Dict, Callable, Tuple
from objax.variable import VarCollection
from objax.gradient import Grad


class HMC(Module):
    def __init__(self,
                 stepsize: Optional[float] = 0.1,
                 burn_in: Optional[int] = 1,
                 thin_factor: Optional[int] = 1,
                 leapfrog_steps: Optional[int] = 0,
                 leapfrog_range: Optional[Tuple[int, int]] = (5, 20),
                 name='HMC'):
        """
        Initialisation parameters for the HMC object

        Args:
            stepsize: The amount that the sampler should move at each successive iteration
            burn_in: The number of samples that should be discarded from the warm-up phase of sampling
            thin_factor: Reduce the autocorrelation within the stationary distribution's samples by only keeping every nth sample, where n here corresponds to the thinning factor value. By default, no thinning is applied.
            leapfrog_steps: How many leapfrog steps should the numerical integrator make. A value of 0 implies that the number of steps should be random in the range [5, 20]
            leapfrog_range: If the leapfrog_steps argument is 0, or less, then the number of leapfrog steps used in the numerical integration is random in the leapfrog_range.
        """
        self.stepsize = stepsize
        self.burn_in = burn_in
        self.thin_factor = thin_factor
        self.leapfrog_steps = leapfrog_steps
        self.leapfrog_random = True if leapfrog_steps < 1 else False
        self.leapfrog_range = leapfrog_range
        self.name = name

    def sample(self,
               n_samples: int,
               key,
               grad_log_posterior: Grad,
               variables: VarCollection,
               timer: Optional[bool] = False):
        """
        Sample from the log-posterior of the Gaussian process with respect to the supplied variables.

        Args:
            n_samples: How many samples should be drawn from the target distribution
            key: A PRNG key for reproducibility
            grad_log_posterior: An ObJax compiled gradient operator e.g., `objax.Grad(posterior.marginal_ll, posterior.vars())`.
            variables: The variables for which we'd like to draw samples with respect to.
            timer: Should a visual timer be returned when sampling to indicate the progress percentage. Default is False.
        """
        samples = []
        n_dims = jnp.sum([v.shape[-1] for v in variables.items()])
        key, *subkeys = jr.split(key, n_samples)
        for idx, skey in enumerate(*subkeys):
            nu = jr.normal(skey, shape=(n_dims, ))
