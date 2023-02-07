<h1 align='center'>JaxKern</h1>
<h2 align='center'>Kernels in Jax.</h2>

[![codecov](https://codecov.io/gh/JaxGaussianProcesses/JaxKern/branch/main/graph/badge.svg?token=8WD7YYMPFS)](https://codecov.io/gh/JaxGaussianProcesses/JaxKern)
[![CircleCI](https://dl.circleci.com/status-badge/img/gh/JaxGaussianProcesses/JaxKern/tree/main.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/gh/JaxGaussianProcesses/JaxKern/tree/main)
[![Documentation Status](https://readthedocs.org/projects/gpjax/badge/?version=latest)](https://gpjax.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/jaxkern.svg)](https://badge.fury.io/py/jaxkern)
[![Downloads](https://pepy.tech/badge/jaxkern)](https://pepy.tech/project/jaxkern)
[![Slack Invite](https://img.shields.io/badge/Slack_Invite--blue?style=social&logo=slack)](https://join.slack.com/t/gpjax/shared_invite/zt-1da57pmjn-rdBCVg9kApirEEn2E5Q2Zw)

## Introduction

JaxKern is Python library for working with kernel functions in JAX. We currently support the following kernels:
* Stationary
    * Radial basis function (Squared exponential)
    * Matérn
    * Powered exponential
    * Rational quadratic
    * White noise
    * Periodic
* Non-stationary
    * Linear 
    * Polynomial
* Non-Euclidean
    * Graph kernels

In addition to this, we implement kernel approximations using the [Random Fourier feature](https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf) approach.

## Example

The following code snippet demonstrates how the first order Matérn kernel can be computed and, subsequently, approximated using random Fourier features.
```python
import jaxkern as jk
import jax.numpy as jnp
import jax.random as jr
key = jr.PRNGKey(123)

# Define the points on which we'll evaluate the kernel
X = jr.uniform(key, shape = (10, 1), minval=-3., maxval=3.)
Y = jr.uniform(key, shape = (20, 1), minval=-3., maxval=3.)

# Instantiate the kernel and its parameters
kernel = jk.Matern32()
params = kernel.init_params(key)

# Compute the 10x10 Gram matrix
Kxx = kernel.gram(params, X)

# Compute the 10x20 cross-covariance matrix
Kxy = kernel.cross_covariance(params, X, Y)

# Build a RFF approximation
approx = RFF(kernel, num_basis_fns = 5)
rff_params = approx.init_params(key)

# Build an approximation to the Gram matrix
Qff = approx.gram(rff_params, X)
```

## Code Structure

All kernels are supplied with a `gram` and `cross_covariance` method. When computing a Gram matrix, there is often some structure in the data (e.g., Markov) that can be exploited to yield a sparse matrix. To instruct JAX how to operate on this, the return type of `gram` is a Linear Operator from [JaxLinOp](https://github.com/JaxGaussianProcesses/JaxLinOp). 

Within [GPJax](https://github.com/JaxGaussianProcesses/GPJax), all kernel computations are handled using JaxKern.

## Documentation

A full set of documentation is a work in progress. However, many of the details in JaxKern can be found in the [GPJax documentation](https://gpjax.readthedocs.io/en/latest/).