# GPJax

GPJax aims to provide a low-level interface to Gaussian process models. Code is written entirely in Jax and Objax to enhance readability, and structured so as to allow researchers to easily extend the code to suit their own needs. When defining GP prior in GPJax, the user need only specify a mean and kernel function. A GP posterior can then be realised by computing the product of our prior with a likelihood function. The idea behind this is that the code should be as close as possible to the maths that we would write on paper when working with GP models.

## Simple example

After importing the necessary dependencies, we'll first simulate some data. 
```python
import jax.numpy as jnp
import jax.random as jr
key = jr.PRNGKey(123)

x = jnp.linspace(-2., 2., 100)
y = jnp.sin(x) + jr.normal(key, shape=x.shape)*0.05
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

# To do

* Spectral kernels ([in progress](https://github.com/thomaspinder/GPJax/tree/spectral))
* Inducing point schemes ([in progress](https://github.com/thomaspinder/GPJax/tree/inducing_points))
* Support for non-conjugate inference