<!-- <h1 align='center'>GPJax</h1>
<h2 align='center'>Gaussian processes in Jax.</h2> -->
<p align="center">
<img width="700" height="300" src="https://github.com/JaxGaussianProcesses/GPJax/raw/master/docs/_static/gpjax_logo.svg" alt="GPJax's logo">
</p>

[![codecov](https://codecov.io/gh/JaxGaussianProcesses/GPJax/branch/master/graph/badge.svg?token=DM1DRDASU2)](https://codecov.io/gh/JaxGaussianProcesses/GPJax)
[![CodeFactor](https://www.codefactor.io/repository/github/jaxgaussianprocesses/gpjax/badge)](https://www.codefactor.io/repository/github/jaxgaussianprocesses/gpjax)
[![Documentation Status](https://readthedocs.org/projects/gpjax/badge/?version=latest)](https://gpjax.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/GPJax.svg)](https://badge.fury.io/py/GPJax)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04455/status.svg)](https://doi.org/10.21105/joss.04455)
[![Downloads](https://pepy.tech/badge/gpjax)](https://pepy.tech/project/gpjax)
[![Slack Invite](https://img.shields.io/badge/Slack_Invite--blue?style=social&logo=slack)](https://join.slack.com/t/gpjax/shared_invite/zt-1da57pmjn-rdBCVg9kApirEEn2E5Q2Zw)

[**Quickstart**](#simple-example)
| [**Install guide**](#installation)
| [**Documentation**](https://gpjax.readthedocs.io/en/latest/)
| [**Slack Community**](https://join.slack.com/t/gpjax/shared_invite/zt-1da57pmjn-rdBCVg9kApirEEn2E5Q2Zw)

GPJax aims to provide a low-level interface to Gaussian process (GP) models in [Jax](https://github.com/google/jax), structured to give researchers maximum flexibility in extending the code to suit their own needs. The idea is that the code should be as close as possible to the maths we write on paper when working with GP models.

# Package support

GPJax was founded by [Thomas Pinder](https://github.com/thomaspinder). Today, the maintenance of GPJax is undertaken by [Thomas Pinder](https://github.com/thomaspinder) and [Daniel Dodd](https://github.com/Daniel-Dodd).

We would be delighted to receive contributions from interested individuals and groups. To learn how you can get involved, please read our [guide for contributing](https://github.com/JaxGaussianProcesses/GPJax/blob/master/CONTRIBUTING.md). If you have any questions, we encourage you to [open an issue](https://github.com/JaxGaussianProcesses/GPJax/issues/new/choose). For broader conversations, such as best GP fitting practices or questions about the mathematics of GPs, we invite you to [open a discussion](https://github.com/JaxGaussianProcesses/GPJax/discussions).

Feel free to join our [Slack Channel](https://join.slack.com/t/gpjax/shared_invite/zt-1da57pmjn-rdBCVg9kApirEEn2E5Q2Zw), where we can discuss the development of GPJax and broader support for Gaussian process modelling.

# Supported methods and interfaces

## Notebook examples

> - [**Conjugate Inference**](https://gpjax.readthedocs.io/en/latest/examples/regression.html)
> - [**Classification with MCMC**](https://gpjax.readthedocs.io/en/latest/examples/classification.html)
> - [**Sparse Variational Inference**](https://gpjax.readthedocs.io/en/latest/examples/uncollapsed_vi.html)
> - [**BlackJax Integration**](https://gpjax.readthedocs.io/en/latest/examples/classification.html)
> - [**Laplace Approximation**](https://gpjax.readthedocs.io/en/latest/examples/classification.html#Laplace-approximation)
> - [**TensorFlow Probability Integration**](https://gpjax.readthedocs.io/en/latest/examples/tfp_integration.html)
> - [**Inference on Non-Euclidean Spaces**](https://gpjax.readthedocs.io/en/latest/examples/kernels.html#Custom-Kernel)
> - [**Inference on Graphs**](https://gpjax.readthedocs.io/en/latest/examples/graph_kernels.html)
> - [**Learning Gaussian Process Barycentres**](https://gpjax.readthedocs.io/en/latest/examples/barycentres.html)
> - [**Deep Kernel Regression**](https://gpjax.readthedocs.io/en/latest/examples/haiku.html)
> - [**Natural Gradients**](https://gpjax.readthedocs.io/en/latest/examples/natgrads.html)

## Guides for customisation
> 
> - [**Custom kernels**](https://gpjax.readthedocs.io/en/latest/examples/kernels.html#Custom-Kernel)
> - [**UCI regression**](https://gpjax.readthedocs.io/en/latest/examples/yacht.html)

## Conversion between `.ipynb` and `.py`
Above examples are stored in [examples](examples) directory in the double percent (`py:percent`) format. Checkout [jupytext using-cli](https://jupytext.readthedocs.io/en/latest/using-cli.html) for more info.

* To convert `example.py` to `example.ipynb`, run:

```bash
jupytext --to notebook example.py
```

* To convert `example.ipynb` to `example.py`, run:

```bash
jupytext --to py:percent example.ipynb
```

# Simple example

Let us import some dependencies and simulate a toy dataset $\mathcal{D}$.

```python
import gpjax as gpx
from jax import grad, jit
import jax.numpy as jnp
import jax.random as jr
import jaxkern as jk
import optax as ox

key = jr.PRNGKey(123)

f = lambda x: 10 * jnp.sin(x)

n = 50
x = jr.uniform(key=key, minval=-3.0, maxval=3.0, shape=(n,1)).sort()
y = f(x) + jr.normal(key, shape=(n,1))
D = gpx.Dataset(X=x, y=y)
```

The function of interest here, $f(\cdot)$, is sinusoidal, but our observations of it have been perturbed by Gaussian noise. We aim to utilise a Gaussian process to try and recover this latent function.

## 1. Constructing the prior and posterior

We begin by defining a zero-mean Gaussian process prior with a radial basis function kernel and assume the likelihood to be Gaussian.

```python
prior = gpx.Prior(kernel = jk.RBF())
likelihood = gpx.Gaussian(num_datapoints = n)
```

Similar to how we would write on paper, the posterior is constructed by the product of our prior with our likelihood.

```python
posterior = prior * likelihood
```

## 2. Learning hyperparameters

Equipped with the posterior, we seek to learn the model's hyperparameters through gradient-optimisation of the marginal log-likelihood. We this below, adding Jax's [just-in-time (JIT)](https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html) compilation to accelerate training. 

```python
mll = jit(posterior.marginal_log_likelihood(D, negative=True))
```

For purposes of optimisation, we'll use optax's Adam.
```
opt = ox.adam(learning_rate=1e-3)
```

We define an initial parameter state through the `initialise` callable.

```python
parameter_state = gpx.initialise(posterior, key=key)
```

Finally, we run an optimisation loop using the Adam optimiser via the `fit` callable.

```python
inference_state = gpx.fit(mll, parameter_state, opt, num_iters=500)
```

## 3. Making predictions

Using our learned hyperparameters, we can obtain the posterior distribution of the latent function at novel test points.

```python
learned_params, _ = inference_state.unpack()
xtest = jnp.linspace(-3., 3., 100).reshape(-1, 1)

latent_distribution = posterior(learned_params, D)(xtest)
predictive_distribution = likelihood(learned_params, latent_distribution)

predictive_mean = predictive_distribution.mean()
predictive_cov = predictive_distribution.covariance()
```

# Installation

## Stable version

The latest stable version of GPJax can be installed via [`pip`](https://pip.pypa.io/en/stable/):

```bash
pip install gpjax
```

> **Note**
>
> We recommend you check your installation version:
> ```python
> python -c 'import gpjax; print(gpjax.__version__)'
> ```



## Development version
> **Warning**
>
> This version is possibly unstable and may contain bugs. 

Clone a copy of the repository to your local machine and run the setup configuration in development mode.
```bash
git clone https://github.com/JaxGaussianProcesses/GPJax.git
cd GPJax
python setup.py develop
```

> **Note**
>
> We advise you create virtual environment before installing:
> ```
> conda create -n gpjax_experimental python=3.10.0
> conda activate gpjax_experimental
>  ```
>
> and recommend you check your installation passes the supplied unit tests:
>
> ```python
> python -m pytest tests/
> ```

# Citing GPJax

If you use GPJax in your research, please cite our [JOSS paper](https://joss.theoj.org/papers/10.21105/joss.04455#).

```
@article{Pinder2022,
  doi = {10.21105/joss.04455},
  url = {https://doi.org/10.21105/joss.04455},
  year = {2022},
  publisher = {The Open Journal},
  volume = {7},
  number = {75},
  pages = {4455},
  author = {Thomas Pinder and Daniel Dodd},
  title = {GPJax: A Gaussian Process Framework in JAX},
  journal = {Journal of Open Source Software}
}
```
