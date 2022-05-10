<h1 align='center'>GPJax</h1>
<h2 align='center'>Gaussian processes in Jax.</h2>

[![codecov](https://codecov.io/gh/thomaspinder/gpjax/branch/master/graph/badge.svg?token=DM1DRDASU2)](https://codecov.io/gh/thomaspinder/gpjax)
[![CodeFactor](https://www.codefactor.io/repository/github/thomaspinder/gpjax/badge)](https://www.codefactor.io/repository/github/thomaspinder/gpjax)
[![Documentation Status](https://readthedocs.org/projects/gpjax/badge/?version=latest)](https://gpjax.readthedocs.io/en/latest/?badge=latest)
[![Downloads](https://pepy.tech/badge/gpjax)](https://pepy.tech/project/gpjax)

[**Quickstart**](#simple-example)
| [**Install guide**](#installation)
| [**Documentation**](https://gpjax.readthedocs.io/en/latest/)

GPJax aims to provide a low-level interface to Gaussian process (GP) models. Code is written entirely in [Jax](https://github.com/google/jax) to enhance readability, and structured to allow researchers to easily extend the code to suit their own needs. When defining GP prior in GPJax, the user need only specify a mean and kernel function. A GP posterior can then be realised by computing the product of our prior with a likelihood function. The idea behind this is that the code should be as close as possible to the maths that we would write on paper when working with GP models.

## Supported methods and interfaces

### Examples

- [**Conjugate Inference**](https://gpjax.readthedocs.io/en/latest/nbs/regression.html)
- [**Classification**](https://gpjax.readthedocs.io/en/latest/nbs/classification.html)
- [**Sparse Variational Inference**](https://gpjax.readthedocs.io/en/latest/nbs/sparse_regression.html)
- [**TensorFlow Probability Integration**](https://gpjax.readthedocs.io/en/latest/nbs/tfp_intergation.html)
- [**Inference on Non-Euclidean Spaces**](https://gpjax.readthedocs.io/en/latest/nbs/kernels.html#Custom-Kernel)
- [**Inference on Graphs**](https://gpjax.readthedocs.io/en/latest/nbs/graph_kernels.html)
- [**Learning Gaussian Process Barycentres**](https://gpjax.readthedocs.io/en/latest/nbs/graph_kernels.html)

### Guides for customisation

- [**Custom Kernel Implementation**](https://gpjax.readthedocs.io/en/latest/nbs/kernels.html#Custom-Kernel)

## Simple example
This simple regression example aims to illustrate the resemblance of GPJax's API with how we write the mathematics of Gaussian processes.

After importing the necessary dependencies, we'll simulate some data.

```python
import gpjax as gpx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.experimental import optimizers
from jax import jit

key = jr.PRNGKey(123)

x = jr.uniform(key=key, minval=-3.0, maxval=3.0, shape=(50,)).sort().reshape(-1, 1)
y = jnp.sin(x) + jr.normal(key, shape=x.shape)*0.05
training = gpx.Dataset(X=x, y=y)
```

The function of interest here is sinusoidal, but our observations of it have been perturbed by independent zero-mean Gaussian noise. We aim to utilise a Gaussian process to try and recover this latent function.

We begin by defining a zero-mean Gaussian process prior with a radial basis function kernel and assume the likelihood to be Gaussian.
```python
prior = gpx.Prior(kernel = gpx.RBF())
likelihood = gpx.Gaussian(num_datapoints = x.shape[0])
```

The posterior is then constructed by the product of our prior with our likelihood.
```python
posterior = prior * likelihood
```

Equipped with the posterior, we proceed to train the model's hyperparameters through gradient-optimisation of the model's marginal log-likelihood. 

We begin by defining a set of initial parameter values through the `initialise` callable. 
```python
params, training_status, constrainer, unconstrainer = gpx.initialise(posterior)
params = gpx.transform(params, unconstrainer)
```

Next, we define the marginal log-likelihood, and Jit compile this to accelerate training. Notice that it is only now that we have incorporated any data into our GP. This is desirable since model building works this way in principle too, where we first define our prior model, then observe some data and use this data to build a posterior.
```python
mll = jit(posterior.marginal_log_likelihood(training, constrainer, negative=True))
```

Finally, we utilise Jax's built-in Adam optimiser and run an optimisation loop.

```python
opt_init, opt_update, get_params = optimizers.adam(step_size=0.01)
opt_state = opt_init(params)

def step(i, opt_state):
    params = get_params(opt_state)
    gradients = jax.grad(mll)(params)
    return opt_update(i, gradients, opt_state)

for i in range(100):
    opt_state = step(i, opt_state)
```

Now that our parameters are optimised, we transform these back to their original constrained space. Using their learned values, we can obtain the posterior distribution of the latent function at a set of novel test points.

```python
final_params = gpx.transform(get_params(opt_state), constrainer)

xtest = jnp.linspace(-3., 3., 100).reshape(-1, 1)

latent_distribution = posterior(training, final_params)(xtest)
predictive_distribution = likelihood(latent_distribution, params)

predictive_mean = predictive_distribution.mean()
predictive_stddev = predictive_distribution.stddev()
```

## Installation

### Stable version

To install the latest stable version of gpjax run

```bash
pip install gpjax
```

### Development version

To install the latest, possibly unstable, version, the following steps should be followed. It is by no means compulsory, but we do advise that you do all of the below inside a virtual environment.

```bash
git clone https://github.com/thomaspinder/GPJax.git
cd GPJax
python setup.py develop
```

It is then recommended that you check your installation using the supplied unit tests

```python
python -m pytest tests/
```
