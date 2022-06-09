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
# GPJax is an incredibly flexible framework due to it's low-level of abstraction. However, by providing a large degree of flexibility to users, it is possbile for pain points to be encountered when developing custom models. In this notebook we try to shed some light on these _gothcas_, providing explanations for their presence and giving appropriate solutions where applicable.
#
# We will not attempt to cover all Jax based edge cases in this notebook as they are covered superbly in the [Jax - The Sharp Bits](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html) notebook.
#
# _This list is non-exhaustive. If you encounter your own confusion, then please consider [ contributing ](https://github.com/thomaspinder/GPJax/blob/master/CONTRIBUTING.md) a PR to this notebook._

# %% [markdown]
# ## Chex Dataclasses
#
# Dataclasses are supported in base Python since version 3.7. However, for a dataclass object to work with Jax, they must be a registered PyTree. This is where [Chex](https://github.com/deepmind/chex).

# %%
import gpjax as gpx
import jax
from chex import dataclass

# %% [markdown]
# ## Parameter transforms
#
# Parameters such as the kernel lengthscale or variance have supports on a subset of the real line. Consequently, as we approach the set's boundary during gradient-based optimisation, it becomes possible that we could step outside of the set's support and consequently introduce a numerical and mathematical error into our model. As an example of this, consider the variance paramater which we know must be strictly positive. At the $t^{\text{th}}$ if we had specify a learning rate
#
# A simple, yet highly impractical, solution to this would be to use tiny learning rates, that way we'd reduce our chance of stepping outside of the parameter's support. However, this does not completely remove the problem and it will also be incredibly costly. An alternative solution is to apply a functional mapping to the parameter that projects it from a constrained subspace of the real line onto the full real line. Gradient updates are then performed on the unconstrained parameter value before its value is projected back onto its original support. Such a transformation is known as a bijection.
#
# In GPJax we supply bijective functions using [Distrax](https://github.com/deepmind/distrax). However, __the user is required to transform parameters within the optimisation routine__ through the following syntax
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
# We make use of this result in GPJax when solving linear systems of equations of the form $Ax = b$. Whilst seemingly abstract at first, such problems are encountered frequently when constructing Gaussian process models. One such example is the marginal log-likelihood
# $$\log p(\y) =  0.5\left(-\y^{\top}\left(\Kff - \sigma_n^2\IdentityMatrix_n \right)^{-1}\y -\log\lvert \Kff + \sigma^2_n\rvert -n\log 2\pi \right)\,.$$
#
# Specifically focussing our attention on the term
# $$\underbrace{\left(\Kff - \sigma_n^2\IdentityMatrix_n \right)}_{\mathbf{A}}^{-1}\y\,,$$
# then we can see that a solution can be found by solving the corresponding system of equations. By working with $\chol(\mathbf{A})$ instead of $\mathbf{A}$ directly though, we can save ourselves a significant amount of floating point operations (flops) by solving two triangular systems of equations (one for $\mathbf{L}$ and another for $\mathbf{L}$) instead of one dense system of equations. Solving two triangular system of equations has complexity $\mathcal{O}(n^3/6)$; a vast improvement compared to regular solvers which have $\mathcal{O}(n^3)$ complexity.
#
# ### The Cholesky drawback
#
# Whilst the computational acceleration given by working with Cholesky factors in place of dense matrices is hopefully now apparent, the _gotcha_ surrounding Cholesky factorisation is the awward numerical instability that can arise due to floating point rounding errors. When we evaluate a covariance function on a set of points that are very _close_ to one another, eigenvalues of the corresponding covariance matrix can get very small. So small in fact that when numerical rounding has been applied the smallest eigenvalues can become negative. Now they are not truly negative, but to our computer they are and this becomes a problem when we want to compute a Cholesky factor as one of the prerequisite requirements for this factorisation is that the input matrix is positive-definite. Clearly, if there are negative eigenvalues then this stipulation has been invalidated.
#
# To resolve this, we apply some numerical jitter to the diagonals of any Gram matrix. Typically this is incredibly small, $10^{-6}$ being the system default, however, for certain problems this jitter amount may need to be increased.

# %% [markdown]
#
