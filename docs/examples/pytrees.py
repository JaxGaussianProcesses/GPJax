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
# `GPJax` **represents all mathematical objects as PyTrees**, giving
#
# - A simple API with a **TensorFlow / PyTorch feel** …
# - … that is **fully compatible** with JAX's functional paradigm.
#
# Our abstraction is based on the Equinox library and aims to offer a Bayesian/GP equivalent to their neural network abstractions. However, we take it a step further by enabling users to create standard Python classes and easily define and modify parameter domains and training statuses for optimisation within a single model object. This object is fully compatible with JAX autogradients without the need for filtering.

# %% [markdown]
# # Gaussian process objects as data:
#
# Within this notebook, we'll be using the squared exponential, or RBF, kernel
# for illustratory purposes. For a pair of vectors $x, y \in \mathbb{R}^d$, its
# form can be mathematically given by
# $$ k(x, y) = \sigma^2\exp\left(\frac{\lVert x-y\rVert_{2}^2}{2\ell^2} \right) $$
# where $\sigma\in\mathbb{R}_{>0}$ is a variance parameter and
# $\ell\in\mathbb{R}_{>0}$ a lengthscale parameter. We call the evaluation of
# $k(x, y)$ the _covariance_.

# %% [markdown]
# ## Dataclasses
#
# A `dataclass` in Python can simplify the creation of classes for storing data and make the code more readable and maintainable. They offer several benefits over a regular class, including:
#
# 1. Conciseness: Dataclasses automatically generate default implementations for several common methods, such as __init__(), __repr__(), and __eq__(), which means less boilerplate code needs to be written.
#
# 2. Type hinting: Dataclasses provide native support for type annotations, which can help catch errors at compile-time and improve code readability.
#
# For the RBF kernel, we use a `dataclass` to represent this object as follows

from dataclasses import dataclass
from jax import Array


@dataclass
class RBF:
    lengthscale: float
    variance: float

    def covariance(self, x: Array, y: Array) -> Array:
        pass


# %% [markdown]
# We have for now left `covariance` empty; however, through this notebook, we shall
# build up to a fully object that can compute covariances in a JAX-compatible way.
#
# For those users who have not seen a `dataclass` before, this statement is equivalent
# to writing


# %%
class RBF:
    def __init__(self, lengthscale: float, variance: float) -> None:
        self.lengthscale = lengthscale
        self.variance = variance

    def covariance(self, x: Array, y: Array) -> Array:
        pass


# %% [markdown]
# However, it a dataclass allows us to significantly reduce the number of lines
# of code needed to represent such objects, particularly as the code's
# complexity increases, as we shall go on to see.
#
# To establish some terminology, within the above RBF `dataclass`, we refer to
# the lengthscale and variance as _fields_. Further, the `RBF.covariance()` is a
# _method_.
#
# - Tom could you perhaps mention the `field` right here?
# - Also feel free to define the covariance here -> maybe demonstrate it is not compatible with JAX out of the box, that leads into the next section - motivation for why we need to talk about PyTree's.

# %% [markdown]
# ## A primer on PyTree’s:
#
# To efficiently represent data JAX provides a `*PyTree*` abstraction. PyTree’s as such, are immutable tree-like structure built out of *‘node’ types* —— container-like Python objects. For instance,
#
# ```python
# [3.14, {"Monte": object(), "Carlo": False}]
# ```
#
# is a PyTree with structure `[*, {"Monte": *, "Carlo": *}]` and leaves `3.14`, `object()`, `False`. As such, most JAX functions operate over pytrees, e.g., `jax.lax.scan`, accepts as input and produces as output a pytrees of JAX arrays.
#
# While the default set of ‘node’ types that are regarded internal pytree nodes is limited to objects such as lists, tuples, and dicts, JAX permits custom types to be readily registered through a global registry, with the values of such traversed recursively (i.e., as a tree!). This is the functionality that we exploit, whereby we construct all Gaussian process models via a tree-structure through our `Module` object

# %% [markdown]
# ## Module
#
# Our design, first and foremost, minimises additional abstractions on top of standard JAX: everything is just PyTrees and transformations on PyTrees, and secondly, provides full compatibility with the main JAX library itself, enhancing integrability with the broader ecosystem of third-party JAX libraries. To achieve this, our core idea is represent all model objects via an immutable tree-structure.
#
#
# ### Defining a Module
#
# There are two main considerations for model parameters, their:
#
# - Trainability status.
# - Domain.
# - Explain why normalising flows don’t break the convention.
# - Mark leaf attributes with `param_field` to set a default bijector and trainable status.
# - Unmarked leaf attributes default to an `Identity` bijector and trainablility set to `True`.
# - Fully compatible with [Distrax](https://github.com/deepmind/distrax) and [TensorFlow Probability](https://www.tensorflow.org/probability) bijectors, so feel free to use these!

# %%
import tensorflow_probability.substrates.jax.bijectors as tfb

from gpjax.base import (
    Module,
    param_field,
)


@dataclass
class RBF(Module):
    lengthscale: float = param_field(1.0, bijector=tfb.Softplus())
    variance: float = param_field(1.0, bijector=tfb.Softplus())

    def covariance(self, x: Array, y: Array) -> Array:
        pass


# %% [markdown]
#
# ### Replacing values
# For consistency with JAX’s functional programming principles, `Module` instances are immutable. And parameters updates occur out-of- place via `replace`.

# %%
kernel = RBF()
kernel = kernel.replace(lengthscale=3.14)  # Update e.g., the lengthscale.
print(kernel)

# %% [markdown]
# ## Transformations 🤖
#
# ### Applying transformations
# Use `constrain` / `unconstrain` to return a `Mytree` with each parameter's bijector `forward` / `inverse` operation applied!

# %%
# Transform kernel to unconstrained space
unconstrained_kernel = kernel.unconstrain()
print(unconstrained_kernel)

# Transform kernel back to constrained space
kernel = unconstrained_kernel.constrain()
print(kernel)

# %% [markdown]
# ### Replacing transformations
# Default transformations can be replaced on an instance via the `replace_bijector` method.

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
# Applying `stop_gradient` **within** the loss function, prevents the flow of gradients during forward or reverse-mode automatic differentiation.
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
# Note this shows any metadata defined via a `dataclasses.field` for the pytree leaves. So feel free to define your own.
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
# It is possible to define your own custom metadata and therefore your own metadata transformations in this vein.
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
