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
#     display_name: base
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 🌳 GPJax PyTrees
#
# `GPJax` **represents all objects as JAX [_PyTrees_](https://jax.readthedocs.io/en/latest/pytrees.html)**, giving
#
# - A simple API with a **TensorFlow / PyTorch feel** …
# - … whilst **fully compatible** with JAX's functional paradigm.
#

# %% [markdown]
# # Gaussian process objects as data:
#
# Our abstraction is based on the Equinox library and aims to offer a
# Bayesian/GP equivalent to their neural network abstractions. However, we take
# it a step further by enabling users to create standard Python classes and
# easily define and modify parameter domains and training statuses for
# optimisation within a single model object. This object is fully compatible
# with JAX autogradients without the need for filtering.
#
# The core idea is to represent all mathematical objects as immutable tree's...
#
# In the following we will consider an academic example, but which should be
# enough to understand the mechanics of how to write custom objects in GPJax.
#

# %% [markdown]
# ## The RBF kernel
#
#
# The kernel in Gaussian process modeling is a mathematical function that
# defines the covariance structure between data points, allowing us to model
# complex relationships and make predictions based on the observed data. The
# radial basis function (RBF, or _squared exponential_) kernel is a popular
# choice. For a pair of vectors $x, y \in \mathbb{R}^d$, its form can be
# mathematically given by
# $$ k(x, y) = \sigma^2\exp\left(\frac{\lVert x-y\rVert_{2}^2}{2\ell^2} \right) $$
# where $\sigma^2\in\mathbb{R}_{>0}$ is a variance parameter and
# $\ell^2\in\mathbb{R}_{>0}$ a lengthscale parameter. Terming the evaluation of
# $k(x, y)$ the _covariance_, we can crudely represent this object in Python as
# follows:

# %%
import jax
import jax.numpy as jnp


class RBF:
    def __init__(self, lengthscale: float, variance: float) -> None:
        self.lengthscale = lengthscale
        self.variance = variance

    def covariance(self, x: jax.Array, y: jax.Array) -> jax.Array:
        l_squared = self.lengthscale
        sigma_sqaured = self.variance

        return sigma_sqaured * jnp.exp(
            -((jnp.linalg.norm(x, y) / 2.0 * l_squared) ** 2)
        )


# %% [markdown]
# However, asserting equivalence between two class instances with exactly the
# same parameters

# %%
kernel_1 = RBF(1.0, 1.0)
kernel_2 = RBF(1.0, 1.0)

print(kernel_1 == kernel_2)

# %% [markdown]
# The assertion is `False`. Yet the lengthscale and variance are certainly the
# same.

# %%
print(kernel_1.lengthscale == kernel_2.lengthscale)
print(kernel_1.variance == kernel_2.variance)

# %% [markdown]
# ## Dataclasses
#
# A `dataclass` in Python can simplify the creation of classes for storing data
# and make the code more readable and maintainable. They offer several benefits
# over a regular class, including:
#
# 1. Conciseness: Dataclasses automatically generate default implementations for
#    several common methods, such as __init__(), __repr__(), and __eq__(), which
#    means less boilerplate code needs to be written.
#
# 2. Type hinting: Dataclasses provide native support for type annotations,
#    which can help catch errors at compile-time and improve code readability.
#
# For the RBF kernel, we use a `dataclass` to represent this object as follows


# %%
from dataclasses import dataclass


@dataclass
class RBF:
    lengthscale: float
    variance: float

    def covariance(self, x: jax.Array, y: jax.Array) -> jax.Array:
        return self.variance * jnp.exp(
            -0.5 * (jnp.linalg.norm(x, y) / self.lengthscale) ** 2
        )


# %% [markdown]
# This time we now have equality between instances.

# %%
kernel_1 = RBF(1.0, 1.0)
kernel_2 = RBF(1.0, 1.0)

print(kernel_1 == kernel_2)

# %% [markdown]
# To establish some terminology, within the above RBF `dataclass`, we refer to
# the lengthscale and variance as _fields_. Further, the `RBF.covariance()` is a
# _method_.

# %% [markdown]
# However, the object we have defined are not yet compatible with JAX, for this
# we must consider PyTree's.

# %% [markdown]
# ## PyTree’s
#
# To efficiently represent data JAX provides a _PyTree_ abstraction. PyTree’s as
# such, are immutable tree-like structure built out of ‘node’ types —
# container-like Python objects. For instance,
#
# ```python
# [3.14, {"Monte": object(), "Carlo": False}]
# ```
#
# is a PyTree with structure `[*, {"Monte": *, "Carlo": *}]` and leaves `3.14`,
# `object()`, `False`. As such, most JAX functions operate over pytrees, e.g.,
# `jax.lax.scan`, accepts as input and produces as output a pytrees of JAX
# arrays.
#
# While the default set of ‘node’ types that are regarded internal pytree nodes
# is limited to objects such as lists, tuples, and dicts, JAX permits custom
# types to be readily registered through a global registry, with the values of
# such traversed recursively (i.e., as a tree!). This is the functionality that
# we exploit, whereby we construct all Gaussian process models via a
# tree-structure through our `Module` object.

# %% [markdown]
# # Module
#
# Our design, first and foremost, minimises additional abstractions on top of
# standard JAX: everything is just PyTrees and transformations on PyTrees, and
# secondly, provides full compatibility with the main JAX library itself,
# enhancing integrability with the broader ecosystem of third-party JAX
# libraries. To achieve this, our core idea is represent all model objects via
# an immutable tree-structure.
#
#
# ### Defining a Module
#
# There are two main considerations for model parameters, their:
#
# - Trainability status.
# - Domain.
# - Explain why normalising flows don’t break the convention.
# - Mark leaf attributes with `param_field` to set a default bijector and
#       trainable status.
# - Unmarked leaf attributes default to an `Identity` bijector and trainablility
#       set to `True`.
# - Fully compatible with [Distrax](https://github.com/deepmind/distrax) and
#       [TensorFlow Probability](https://www.tensorflow.org/probability) bijectors,
#       so feel free to use these!

# %%
import tensorflow_probability.substrates.jax.bijectors as tfb
from gpjax import base


@dataclass
class RBF(base.Module):
    lengthscale: float = base.param_field(1.0, bijector=tfb.Softplus(), trainable=True)
    variance: float = base.param_field(1.0, bijector=tfb.Softplus(), trainable=True)

    def covariance(self, x: jax.Array, y: jax.Array) -> jax.Array:
        return self.variance * jnp.exp(
            -0.5 * (jnp.linalg.norm(x, y) / self.lengthscale) ** 2
        )


# %% [markdown]
#
# ### Replacing values
# For consistency with JAX’s functional programming principles, `Module`
# instances are immutable. Parameter updates can be achieved out-of-place via
# `replace`.

# %%
kernel = RBF()
kernel = kernel.replace(lengthscale=3.14)  # Update e.g., the lengthscale.
print(kernel)

# %% [markdown]
# ## Transformations 🤖
#
# ### Applying transformations
# Use `constrain` / `unconstrain` to return a `Mytree` with each parameter's
# bijector `forward` / `inverse` operation applied!

# %%
# Transform kernel to unconstrained space
unconstrained_kernel = kernel.unconstrain()
print(unconstrained_kernel)

# Transform kernel back to constrained space
kernel = unconstrained_kernel.constrain()
print(kernel)

# %% [markdown]
# ### Replacing transformations
# Default transformations can be replaced on an instance via the
# `replace_bijector` method.

# %%
new_kernel = kernel.replace_bijector(lengthscale=tfb.Identity())

# Transform kernel to unconstrained space
unconstrained_kernel = new_kernel.unconstrain()
print(unconstrained_kernel)

# Transform kernel back to constrained space
new_kernel = unconstrained_kernel.constrain()
print(new_kernel)

# %% [markdown]
# ## Trainability 🚂
#
# ### Applying trainability
#
# Applying `stop_gradient` **within** the loss function, prevents the flow of
# gradients during forward or reverse-mode automatic differentiation.
# ```python
# import jax
#
# # Create simulated data.
# n = 100
# key = jax.random.PRNGKey(123)
# x = jax.random.uniform(key, (n, ))
# y = 3.0 * x + 2.0 + 1e-3 * jax.random.normal(key, (n, ))
#
#
# # Define a mean-squared-error loss.
# def loss(model: SimpleModel) -> float:
#    model = model.stop_gradient() # 🛑 Stop gradients!
#    return jax.numpy.sum((y - model(x))**2)
#
# jax.grad(loss)(model)
# ```
# ```
# SimpleModel(weight=0.0, bias=-188.37418)
# ```
# As `weight` trainability was set to `False`, it's gradient is zero as expected!
#
# ### Replacing trainability
# Default trainability status can be replaced via the `replace_trainable` method.
# ```python
# new = model.replace_trainable(weight=True)
# jax.grad(loss)(model)
# ```
# ```
# SimpleModel(weight=-121.42676, bias=-188.37418)
# ```
# And we see that `weight`'s gradient is no longer zero.
#
# ## Metadata
#
# ### Viewing `field` metadata
# View field metadata pytree via `meta`.
# ```python
# from mytree import meta
# meta(model)
# ```
# ```
# SimpleModel(weight=({'bijector': Bijector(forward=<function <lambda> at 0x17a024e50>, inverse=<function <lambda> at 0x17a024430>), 1.0), 'trainable': False, 'pytree_node': True}, bias=({}, 2.0))
# ```
#
# Or the metadata pytree leaves via `meta_leaves`.
# ```python
# from mytree import meta_leaves
# meta_leaves(model)
# ```
# ```
# [({}, 2.0),
#  ({'bijector': Bijector(forward=<function <lambda> at 0x17a024e50>, inverse=<function <lambda> at 0x17a024430>),
#   'trainable': False,
#   'pytree_node': True}, 1.0)]
# ```
# Note this shows any metadata defined via a `dataclasses.field` for the pytree
# leaves. So feel free to define your own.
#
# ### Applying `field` metadata
# Leaf metadata can be applied via the `meta_map` function.
# ```python
# from mytree import meta_map
#
# # Function passed to `meta_map` has its argument as a `(meta, leaf)` tuple!
# def if_trainable_then_10(meta_leaf):
#     meta, leaf = meta_leaf
#     if meta.get("trainable", True):
#         return 10.0
#     else:
#         return leaf
#
# meta_map(if_trainable_then_10, model)
# ```
# ```
# SimpleModel(weight=1.0, bias=10.0)
# ```
# It is possible to define your own custom metadata and therefore your own
# metadata transformations in this vein.
#
# ## Static fields
# Fields can be marked as static via simple_pytree's `static_field`.
#
# ```python
# import jax.tree_util as jtu
# from simple_pytree import static_field
#
# class StaticExample(Mytree):
#     b: float = static_field
#
#     def __init__(self, a=1.0, b=2.0):
#         self.a=a
#         self.b=b
#
# jtu.tree_leaves(StaticExample())
# ```
# ```
# [1.0]
# ```
