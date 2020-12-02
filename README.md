# GPJax

Aims to providers researchers with the necessary objects to develop new Gaussian process model. The aim of this package is to enable one to code a GP how you would write it on paper. 

GPJax is written entirely in Jax and follows a more functional framework to modelling.

## Simple example

After importing the necessary dependencies, we'll first simulate some data. 
```python
import jax.numpy as jnp
import jax.random as jnr
key = jnr.PRNGKey(123)

x = jnp.linspace(-2., 2., 100)
y = jnp.sin(x) + jnr.normal(key, shape=x.shape)*0.05
```

As can be seen, the latent function of interest here is a sinusoidal function. However, it has been perturbed by some zero-mean Gaussian noise with variance of 0.05. We can use a Gaussian process model to try and recover this latent function.

```python
from gpjax import Prior, RBF

kernel = RBF(lengthscale=1.0, variance=1.0)
f = Prior(kernel)
```
 
In the presence of a likelihood function which we'll here assume to be Gaussian, we can optimise the marginal log-likelihood of the Gaussian process prior multiplied by the likelihood to obtain a posterior distribution over the latent function.

```python
from gpjax import Gaussian

likelihood = Gaussian()
posterior = f * likelihood
```

## Installation

To 