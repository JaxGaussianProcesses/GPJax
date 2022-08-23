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

GPJax aims to provide a low-level interface to Gaussian process (GP) models in [Jax](https://github.com/google/jax), structured to give researchers maximum flexibility in extending the code to suit their own needs. We define a GP prior in GPJax by specifying a mean and kernel function and multiply this by a likelihood function to construct the posterior. The idea is that the code should be as close as possible to the maths we write on paper when working with GP models.

## Package support

GPJax was created by [Thomas Pinder](https://github.com/thomaspinder). Today, the maintenance of GPJax is undertaken by Thomas and [Daniel Dodd](https://github.com/Daniel-Dodd).

We would be delighted to review pull requests (PRs) from new contributors. Before contributing, please read our [guide for contributing](https://github.com/thomaspinder/GPJax/blob/master/CONTRIBUTING.md). If you do not have the capacity to open a PR, or you would like guidance on how best to structure a PR, then please [open an issue](https://github.com/thomaspinder/GPJax/issues/new/choose). For broader discussions on best practices for fitting GPs, technical questions surrounding the mathematics of GPs, or anything else that you feel doesn't quite constitue an issue, please start a discussion thread in our [discussion tracker](https://github.com/thomaspinder/GPJax/discussions).

We have recently set up a Slack channel where we hope to facilitate discussions around the development of GPJax and broader support for Gaussian process modelling. If you'd like to join the channel, then please follow [this invitation link](https://join.slack.com/t/gpjax/shared_invite/zt-1da57pmjn-rdBCVg9kApirEEn2E5Q2Zw) which will take you to the GPJax Slack community.

## Supported methods and interfaces

### Examples

- [**Conjugate Inference**](https://gpjax.readthedocs.io/en/latest/nbs/regression.html)
- [**Classification with MCMC**](https://gpjax.readthedocs.io/en/latest/nbs/classification.html)
- [**Sparse Variational Inference**](https://gpjax.readthedocs.io/en/latest/nbs/uncollapsed_vi.html)
- [**BlackJax Integration**](https://gpjax.readthedocs.io/en/latest/nbs/classification.html)
- [**Laplace Approximations**](https://gpjax.readthedocs.io/en/latest/nbs/classification.html#Laplace-approximation)
- [**TensorFlow Probability Integration**](https://gpjax.readthedocs.io/en/latest/nbs/tfp_integration.html)
- [**Inference on Non-Euclidean Spaces**](https://gpjax.readthedocs.io/en/latest/nbs/kernels.html#Custom-Kernel)
- [**Inference on Graphs**](https://gpjax.readthedocs.io/en/latest/nbs/graph_kernels.html)
- [**Learning Gaussian Process Barycentres**](https://gpjax.readthedocs.io/en/latest/nbs/graph_kernels.html)
- [**Deep Kernel Regression**](https://gpjax.readthedocs.io/en/latest/nbs/haiku.html)

### Guides for customisation

- [**Custom Kernel Implementation**](https://gpjax.readthedocs.io/en/latest/nbs/kernels.html#Custom-Kernel)

## Simple example

This simple regression example aims to illustrate the resemblance of GPJax's API with how we write the mathematics of Gaussian processes.

After importing the necessary dependencies, we'll simulate some data.

```python
import gpjax as gpx
from jax import grad, jit
import jax.numpy as jnp
import jax.random as jr
import optax as ox

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

Equipped with the posterior, we proceed to train the model's hyperparameters through gradient-optimisation of the marginal log-likelihood.

We begin by defining a set of initial parameter values through the `initialise` callable.

```python
parameter_state = gpx.initialise(posterior, key=key)
params, trainables, constrainer, unconstrainer = parameter_state.unpack()
params = gpx.transform(params, unconstrainer)
```

Next, we define the marginal log-likelihood, adding Jax's just-in-time (JIT) compilation to accelerate training. Notice that this is the first instance of incorporating data into our model. Model building works this way in principle too, where we first define our prior model, then observe some data and use this data to build a posterior.

```python
mll = jit(posterior.marginal_log_likelihood(training, constrainer, negative=True))
```

Finally, we utilise Jax's built-in Adam optimiser and run an optimisation loop.

```python
opt = ox.adam(learning_rate=0.01)
opt_state = opt.init(params)

for _ in range(100):
  grads = grad(mll)(params)
  updates, opt_state = opt.update(grads, opt_state)
  params = ox.apply_updates(params, updates)
```

Now that our parameters are optimised, we transform these back to their original constrained space. Using their learned values, we can obtain the posterior distribution of the latent function at novel test points.

```python
final_params = gpx.transform(params, constrainer)

xtest = jnp.linspace(-3., 3., 100).reshape(-1, 1)

latent_distribution = posterior(training, final_params)(xtest)
predictive_distribution = likelihood(latent_distribution, params)

predictive_mean = predictive_distribution.mean()
predictive_stddev = predictive_distribution.stddev()
```

## Installation

### Stable version

To install the latest stable version of GPJax run

```bash
pip install gpjax
```

### Development version

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
