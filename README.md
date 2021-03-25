# GPJax

[![codecov](https://codecov.io/gh/thomaspinder/gpjax/branch/master/graph/badge.svg?token=DM1DRDASU2)](https://codecov.io/gh/thomaspinder/gpjax)
[![CodeFactor](https://www.codefactor.io/repository/github/thomaspinder/gpjax/badge)](https://www.codefactor.io/repository/github/thomaspinder/gpjax)
[![Documentation Status](https://readthedocs.org/projects/gpjax/badge/?version=latest)](https://gpjax.readthedocs.io/en/latest/?badge=latest)

[**Quickstart**](#simple-example)
| [**Install guide**](#installation)
| [**Documentation**](https://gpjax.readthedocs.io/en/latest/)

GPJax aims to provide a low-level interface to Gaussian process models. Code is written entirely in [Jax](https://github.com/google/jax) to enhance readability, and structured to allow researchers to easily extend the code to suit their own needs. When defining GP prior in GPJax, the user need only specify a mean and kernel function. A GP posterior can then be realised by computing the product of our prior with a likelihood function. The idea behind this is that the code should be as close as possible to the maths that we would write on paper when working with GP models.

## Supported methods and interfaces

* [__Conjugate Inference__](https://gpjax.readthedocs.io/en/latest/nbs/regression.html)
* [__Classification__](https://gpjax.readthedocs.io/en/latest/nbs/classification.html) through [__Hamiltonian Monte Carlo__](https://gpjax.readthedocs.io/en/latest/nbs/numpyro_interface.html#Hamiltonian-Monte-Carlo) or [__Variational Inference__](https://gpjax.readthedocs.io/en/latest/nbs/numpyro_interface.html#Approximate-Inference-with-Guides)
* [__Spectral Kernels__](https://gpjax.readthedocs.io/en/latest/nbs/spectral.html)
* [__NumPyro Integraion__](https://gpjax.readthedocs.io/en/latest/nbs/numpyro_interface.html#)
* [__TensorFlow Probability Integration__](https://gpjax.readthedocs.io/en/latest/nbs/tfp_interface.html)

## Simple example

After importing the necessary dependencies, we'll first simulate some data. 
```python
import gpjax
from gpjax import Dataset
import jax
import jax.numpy as jnp
import jax.random as jr
key = jr.PRNGKey(123)

x = jr.uniform(key=key, minval=-3.0, maxval=3.0, shape=(50,)).sort().reshape(-1, 1)
y = jnp.sin(x) + jr.normal(key, shape=x.shape)*0.05
training = Dataset(X=x, y=y)
```

As can be seen, the latent function of interest here is a sinusoidal function. However, it has been perturbed by some zero-mean Gaussian noise with variance of 0.05. We can use a Gaussian process model to try and recover this latent function.

```python
from gpjax.kernels import RBF
from gpjax.gps import Prior

f = Prior(kernel = RBF())
```
 
In the presence of a likelihood function which we'll here assume to be Gaussian, we can optimise the marginal log-likelihood of the Gaussian process prior multiplied by the likelihood to obtain a posterior distribution over the latent function.

```python
from gpjax.likelihoods import Gaussian

likelihood = Gaussian()
posterior = f * likelihood
```

Equipped with the Gaussian process posterior, we can now optimise the model's hyperparameters (note, we need not optimise the latent function here due to the Gaussian conjugacy.). To do this, we can either define our parameters by hand through a dictionary, or realise a set of default parameters through the `initialise` callable. For brevity, we'll do the latter here but see the [regression notebook](https://github.com/thomaspinder/GPJax/blob/master/docs/nbs/regression.ipynb) for a full discussion on parameter initialisation and transformation. 

```python
from gpjax.parameters import initialise, build_all_transforms
from gpjax.config import get_defaults

params = initialise(posterior)
configs = get_defaults()
constrainer, unconstrainer = build_all_transforms(params.keys(), configs)
params = unconstrainer(params)
```

With initial values defined, we can now optimise the hyperparameters' value by carrying out gradient-based optimisation with respect to the GP's marginal log-likelihood. We'll do this now using Jax's built in optimisers, namely the Adam optimiser with a step-size of 0.01. We can also Jit compile our objective function to accelerate training. You'll notice that it is only now that we have incorporated any data into our GP. This is desirable, as this is exactly how model building works in principle too, where we first build our prior model, then observe some data and use this data to build a posterior.

```python
from gpjax.objectives import marginal_ll
from jax.experimental import optimizers

mll = jit(marginal_ll(posterior, transform=constrainer, negative=True))

opt_init, opt_update, get_params = optimizers.adam(step_size=0.01)
opt_state = opt_init(params)
def step(i, opt_state):
    p = get_params(opt_state)
    g = jax.grad(mll)(p, training)
    return opt_update(i, g, opt_state)


for i in range(100):
    opt_state = step(i, opt_state)
```


Our parameters are now optimised. We can retransfrom these back onto the parameter's original constrained space and, using this learned value, query the GP at a set of test points.

```python
from gpjax.predict import mean, variance


final_params = constrainer(get_params(opt_state))

xtest = jnp.linspace(-3., 3., 100).reshape(-1, 1)

predictive_mean = mean(posterior, final_params, training)(xtest)
predictive_variance = variance(posterior, final_params, training)(xtest)
```

## Installation

### Stable version

To install the latest stable version of gpjax run
```bash
pip install gpjax
```

### Development version

To install the lastest, possibly unstable, version, the following steps should be followed. It is by no means compulsory, but we do advise that you do all of the below inside a virtual environment.

```bash
git clone https://github.com/thomaspinder/GPJax.git
cd GPJax 
python setup.py develop
```

It is then recommended that you check your installation using the supplied unit tests
```python
python -m pytest tests/
```

**Note that installing the latest version of GPJax on Apple M1 devices is currently unstable.**
