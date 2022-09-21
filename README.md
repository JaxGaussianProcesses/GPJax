<!-- <h1 align='center'>GPJax</h1>
<h2 align='center'>Gaussian processes in Jax.</h2> -->
<p align="center">
<img width="700" height="300" src="https://github.com/thomaspinder/GPJax/raw/master/docs/_static/gpjax_logo.svg" alt="GPJax's logo">
</p>

[![codecov](https://codecov.io/gh/thomaspinder/gpjax/branch/master/graph/badge.svg?token=DM1DRDASU2)](https://codecov.io/gh/thomaspinder/gpjax)
[![CodeFactor](https://www.codefactor.io/repository/github/thomaspinder/gpjax/badge)](https://www.codefactor.io/repository/github/thomaspinder/gpjax)
[![Documentation Status](https://readthedocs.org/projects/gpjax/badge/?version=latest)](https://gpjax.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/GPJax.svg)](https://badge.fury.io/py/GPJax)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04455/status.svg)](https://doi.org/10.21105/joss.04455)
[![Downloads](https://pepy.tech/badge/gpjax)](https://pepy.tech/project/gpjax)
[![Slack Invite](https://img.shields.io/badge/Slack_Invite--blue?style=social&logo=slack)](https://join.slack.com/t/gpjax/shared_invite/zt-1da57pmjn-rdBCVg9kApirEEn2E5Q2Zw)

[**Quickstart**](#simple-example)
| [**Install guide**](#installation)
| [**Documentation**](https://gpjax.readthedocs.io/en/latest/)
| [**Slack Community**](https://join.slack.com/t/gpjax/shared_invite/zt-1da57pmjn-rdBCVg9kApirEEn2E5Q2Zw)

`GPJax` is a didactic Gaussian process library that supports GPU acceleration and just-in-time compilation thanks to its [`Jax`](https://github.com/google/jax) backend. We seek to provide a flexible API as close as possible to how the underlying mathematics is written on paper.

## Package support

`GPJax` was created by [Thomas Pinder](https://github.com/thomaspinder). Today, maintenance is undertaken by Thomas and [Daniel Dodd](https://github.com/Daniel-Dodd).

Everyone can contribute to `GPJax`, and we value everyone’s contributions. There are several ways to contribute, including:

We welcome pull requests (PRs) from new contributors. Before contributing, please read our [guide for contributing](https://github.com/thomaspinder/GPJax/blob/master/CONTRIBUTING.md). If you have any questions, feel free to [open an issue](https://github.com/thomaspinder/GPJax/issues/new/choose). For broader discussions, such as best GP fitting practices, technical questions surrounding the mathematics of GPs, we encourage you to [open a discussion](https://github.com/thomaspinder/GPJax/discussions).

Feel free to chat to us on [Slack](https://join.slack.com/t/gpjax/shared_invite/zt-1da57pmjn-rdBCVg9kApirEEn2E5Q2Zw) where we can discuss the development of GPJax and broader support for Gaussian process modelling.

## Supported methods and interfaces

#### Examples

- [**Conjugate Inference**](https://gpjax.readthedocs.io/en/latest/examples/regression.html)
- [**Classification with MCMC**](https://gpjax.readthedocs.io/en/latest/examples/classification.html)
- [**Sparse Variational Inference**](https://gpjax.readthedocs.io/en/latest/examples/uncollapsed_vi.html)
- [**BlackJax Integration**](https://gpjax.readthedocs.io/en/latest/examples/classification.html)
- [**Laplace Approximations**](https://gpjax.readthedocs.io/en/latest/examples/classification.html#Laplace-approximation)
- [**TensorFlow Probability Integration**](https://gpjax.readthedocs.io/en/latest/examples/tfp_integration.html)
- [**Inference on Non-Euclidean Spaces**](https://gpjax.readthedocs.io/en/latest/examples/kernels.html#Custom-Kernel)
- [**Inference on Graphs**](https://gpjax.readthedocs.io/en/latest/examples/graph_kernels.html)
- [**Learning Gaussian Process Barycentres**](https://gpjax.readthedocs.io/en/latest/examples/barycentres.html)
- [**Deep Kernel Regression**](https://gpjax.readthedocs.io/en/latest/examples/haiku.html)

#### Guides for customisation

- [**Custom Kernel Implementation**](https://gpjax.readthedocs.io/en/latest/examples/kernels.html#Custom-Kernel)

## Simple example

Let us import some dependencies and simulate a toy dataset $\mathcal{D}$.

```python
import gpjax as gpx
from jax import grad, jit
import jax.numpy as jnp
import jax.random as jr
import optax as ox

key = jr.PRNGKey(123)

f = lambda x: 10 * jnp.sin(x)

n = 50
x = jr.uniform(key=key, minval=-3.0, maxval=3.0, shape=(n,1)).sort()
y = f(x) + jr.normal(key, shape=(n,1))
D = gpx.Dataset(X=x, y=y)
```

The function of interest here, $f(\cdot)$, is sinusoidal, but our observations of it have been perturbed by Gaussian noise. We aim to utilise a Gaussian process to try and recover this latent function.

#### 1. Constructing the prior and posterior

We begin by defining a zero-mean Gaussian process prior with a radial basis function kernel and assume the likelihood to be Gaussian.

```python
prior = gpx.Prior(kernel = gpx.RBF())
likelihood = gpx.Gaussian(num_datapoints = n)
```

Similar to how we would write on paper, the posterior is constructed by the product of our prior with our likelihood.

```python
posterior = prior * likelihood
```

#### 2. Learning hyperparameters

Equipped with the posterior, we seek to learn the model's hyperparameters through gradient-optimisation of the marginal log-likelihood. We this below, adding Jax's [just-in-time (JIT)](https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html) compilation to accelerate training. 

```python
mll = jit(posterior.marginal_log_likelihood(training, negative=True))
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
inference_state = gpx.fit(mll, parameter_state, opt, n_iters=500)
```

#### 3. Making predictions

Using our learned hyperparameters, we can obtain the posterior distribution of the latent function at novel test points.

```python
learned_params, _ = inference_state.unpack()
xtest = jnp.linspace(-3., 3., 100).reshape(-1, 1)

latent_distribution = posterior(training, learned_params)(xtest)
predictive_distribution = likelihood(latent_distribution, params)

predictive_mean = predictive_distribution.mean()
predictive_stddev = predictive_distribution.stddev()
```

## Installation

#### Stable version

To install the latest stable version of GPJax run

```bash
pip install gpjax
```

#### Development version

To install the latest (possibly unstable) version, the following steps should be followed. It is by no means compulsory, but we do advise that you do all of the below inside a virtual environment.

```bash
git clone https://github.com/thomaspinder/GPJax.git
cd GPJax
python setup.py develop
```

We then recommend you check your installation using the supplied unit tests.

```python
python -m pytest tests/
```

## Citing GPJax

If you use GPJax in your research, please cite our [JOSS paper](https://joss.theoj.org/papers/10.21105/joss.04455#). Sample Bibtex is given below:

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
