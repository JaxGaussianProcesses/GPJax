# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3.9.7 ('gpjax')
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Common Gotchas in GPJax
#
# GPJax is incredibly flexible due to it's low-level of abstraction. However, by providing a significant degree of flexibility to users, pain points can be encountered when developing custom models. In this notebook, we try to shed some light on these, _gothcas_, providing explanations for their presence and giving appropriate solutions where applicable.
#
# We will not attempt to cover all Jax-based edge cases in this notebook as they are covered superbly in the [Jax - The Sharp Bits](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html) notebook.
#
# _This list is non-exhaustive. If you encounter confusion, please consider opening [contributing](https://github.com/thomaspinder/GPJax/blob/master/CONTRIBUTING.md) a PR to this notebook._

# %% [markdown]
# ## Chex Dataclasses
#
# Dataclasses have been supported in base Python since version 3.7. However, for dataclass objects to work with Jax, they must be a registered PyTree. This is where [Chex](https://github.com/deepmind/chex) comes in. Chex is a Jax library, that provides an implementation of Python's `dataclass` object that is a valid PyTree. Using dataclasses in place of regular Python classes removes a significant amount of boilerplate when defining new and thus keeps the underlying codebase more concise. However, there are a few specifics that one should be mindful of when using dataclasses.
#
# ### Instantiation
#
# When instantiating a dataclass, a requirement is that all arguments are keyword arguments. As an example, consider the instantiation of a GP prior, parameterised by an RBF kernel:
# ```diff
# ❌ prior = gpx.Prior(gpx.RBF())
# ✅ prior = gpx.Prior(kernel=gpx.RBF())
# ```
# Running the former will result in an error because the kernel argument is missing its associated keyword. Takeaway: __when instantiating a GPJax object, all arguments must be named arguments__.
#
# ### Inheritance
#
# Dataclasses are just regular classes under the hood. However, there is some nuance when inheriting dataclasses from one another. Namely, this boils down to the inheritance order being in reverse.
#
# Like regular classes and functions, when defining a custom dataclass, non-default arguments must precede default valued arguments. However, when inheriting dataclasses, this process can be a little subtle. For example, imagine the following case where we have a generic stationary kernel
# ```python
# @dataclass
# class Stationary:
#     lengthscale: float
#     variance: float = 1.
# ```
# If we now wish to inherit the `Stationary` dataclass to define a Matérn kernel, we may desire to make the smoothness parameter an additional field. However, the following code would error
# ```python
# @dataclass
# class Matern(Stationary):
#     nu: float
# ```
# due to the inheritence order of dataclass fields. If we unravel this inheritance, then in this example, Python is trying to create the following class:
# ```python
# @dataclass
# class Matern:
#     lengthscale: float
#     variance: float = 1.
#     nu: float
# ```
# Hopefully, now it is clear why this has errored - having the default `variance` field precede the required argument `nu` goes against Python's design structure. So how can this be resolved?
# __Option 1:__ Multiple inheritance is one option where we could inherit multiple classes to create a new class. This can be a clean solution at times and we do make use of this in GPJax when defining [Graph kernels](https://gpjax.readthedocs.io/en/latest/nbs/graph_kernels.html). However, sometimes, as is the case here, the abstractions can quickly become quite clunky. For this example, one would have to do something to the effect of
# ```python
# @dataclass
# class _Smoothness:
#     nu: float
#
# @dataclass
# class Matern(Stationary, _Smoothness):
#     pass
# ```
# Dataclasses are inherited in reverse order, so we have now resolved the field ordering such that they are now 
# ```python
# @dataclass
# class Matern:
#     nu: float
#     lengthscale: float
#     variance: float = 1.
# ```
# However, the downside is that we have this awkward `_Smoothness` object in our code.
#
# __Option 2:__ Think carefully about which arguments should have default options. Another option is to make the `variance` field a required argument. This way, the inheritance order does not matter as all arguments will be required.
#
# ####

# %% [markdown]
# ## Parameter transforms
#
# Parameters such as the kernel's length-scale or variance have their supports on a constrained subset of the real-line. Consequently, as we approach the set's boundary during gradient-based optimisation, it becomes possible that we could step outside of the set's support and introduce a numerical and mathematical error into our model. For example, consider the variance parameter $\sigma$, which we know must be strictly positive. If at the $t^{\text{th}}$ iteration, our current estimate of $\sigma$ was $0.03$ and our derivative informed us that $\sigma$ should decrease, if our learning rate is greater is than $0.03$, then we would end up with a negative variance term.
#
 #A simple but impractical solution would be to use tiny learning rates, which would reduce the possibility of stepping outside of the parameter's support. However, this would be incredibly costly and does not eradicate the problem. An alternative solution is to apply a functional mapping to the parameter that projects it from a constrained subspace of the real-line onto the full real-line. Here, gradient updates are applied in the unconstrained parameter space before transforming the value back to the original support of the parameters. Such a transformation is known as a bijection.
#
# In GPJax, we supply bijective functions using [Distrax](https://github.com/deepmind/distrax). However, __the user is required to transform parameters within the optimisation routine__ through the following syntax
# ```python
# unconstrained_params = gpx.transform(constrained_params, unconstrainers)
# ```
# The constrained parameters and associated bijections are returned when a user initialises their model using `gpx.initialise`.

# %% [markdown]
# ## Cholesky factors
#
# > "_Symmetric positive definiteness is one of the highest accolades to which a matrix can aspire_" - Nicholas Highman, Accuracy and stability of numerical algorithms
#
#
# ### Why should we care about Cholesky factors?
#
# The covariance matrix of a kernel is a symmetric positive definite matrix. As such, we have a range of tools at our disposal to make subsequent operations on the covariance matrix faster. One of these tools is the Cholesky factorisation that uniquely decomposes any symmetric positive-definite matrix $\Sigma$ by
# $$\Sigma = \mathbf{L}\mathbf{L}^{\top}$$
# where $\mathbf{L}$ is a lower-triangular matrix.
#
# We make use of this result in GPJax when solving linear systems of equations of the form $Ax = b$. Whilst seemingly abstract at first, such problems are frequently encountered when constructing Gaussian process models. One such example is the marginal log-likelihood
# $$\log p(\y) =  0.5\left(-\y^{\top}\left(\Kff - \sigma_n^2\IdentityMatrix_n \right)^{-1}\y -\log\lvert \Kff + \sigma^2_n\rvert -n\log 2\pi \right)\,.$$
#
# Specifically focussing our attention on the term
# $$\underbrace{\left(\Kff - \sigma_n^2\IdentityMatrix_n \right)^{-1}}_{\mathbf{A}}\y\,,$$
# then we can see a solution can be optained by solving the corresponding system of equations. By working with $\chol{\mathbf{A}}$ instead of $\mathbf{A}$ though, we can save ourselves a significant amount of floating-point operations (flops) by solving two triangular systems of equations (one for $\mathbf{L}$ and another for $\mathbf{L}$) instead of one dense system of equations. Solving two triangular systems of equations has complexity $\mathcal{O}(n^3/6)$; a vast improvement compared to regular solvers that have $\mathcal{O}(n^3)$ complexity.
#
# ### The Cholesky drawback
#
# While the computational acceleration provided by using Cholesky factors instead of dense matrices is hopefully now apparent, an awkward numerical instability _gotcha_ can arise due to floating-point rounding errors. When we evaluate a covariance function on a set of points that are very _close_ to one another, eigenvalues of the corresponding covariance matrix can get very small. So small that after numerical rounding, the smallest eigenvalues can become negative-valued. While not truly less than zero, our computer thinks they are, which becomes a problem when we want to compute a Cholesky factor since this requires that the input matrix is positive-definite. If there are negative eigenvalues, then this stipulation has been invalidated.
#
# To resolve this, we apply some numerical _jitter_ to the diagonals of any Gram matrix. Typically this is incredibly small, with $10^{-6}$ being the system default. However, for some problems, this amount may need to be increased. 

# %% [markdown]
#
