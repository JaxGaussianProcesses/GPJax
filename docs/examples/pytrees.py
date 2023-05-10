# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: title,-all
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
# # 🌳 GPJax Module
#
# `GPJax` **represents all objects as JAX
# [_PyTrees_](https://jax.readthedocs.io/en/latest/pytrees.html)**, giving
#
# - A simple API with a **TensorFlow / PyTorch feel** ...
# - ... whilst **fully compatible** with JAX's functional paradigm ...
# - ... And **works out of the box** (no filtering) with JAX's transformations
#   such as `grad`.
#
# We achieve this through providing a base `Module` abstraction to cleanly
# handles parameter trainability and optimising transformations for JAX models.
#

# %% [markdown]
# # Gaussian process objects as data
#
# Our abstraction is inspired by the Equinox library and aims to offer a
# Bayesian/Gaussian process equivalent to their neural network abstractions. Our
# approach goes further by allowing users to easily create Python classes and
# define parameter domains and training statuses for optimisation within a
# single model object. This object can be used with JAX autogradients without
# any filtering.
#
# The fundamental concept is to describe every model object as an immutable tree
# structure, where every method is a function of the state (represented by the
# tree's leaves).
#
# To help you understand how to create custom objects in GPJax, we will look at
# an academic example in the following section.

# %% [markdown]
# ## The RBF kernel
#
#
# The kernel in Gaussian process modeling is a mathematical function that
# defines the covariance structure between data points, allowing us to model
# complex relationships and make predictions based on the observed data. The
# radial basis function (RBF, or _squared exponential_) kernel is a popular
# choice. For a pair of vectors $x, y \in \mathbb{R}^d$, its form can be
# mathematically given by $$ k(x, y) = \sigma^2\exp\left(\frac{\lVert
# x-y\rVert_{2}^2}{2\ell^2} \right) $$ where $\sigma^2\in\mathbb{R}_{>0}$ is a
# variance parameter and $\ell^2\in\mathbb{R}_{>0}$ a lengthscale parameter.
# Terming the evaluation of $k(x, y)$ the _covariance_, we can crudely represent
# this object as a Python `dataclass` as follows:

# %%
import jax
import jax.numpy as jnp
from dataclasses import dataclass, field


@dataclass
class RBF:
    lengthscale: float = field(default=1.0)
    variance: float = field(default=1.0)

    def covariance(self, x: float, y: float) -> jax.Array:
        return self.variance * jnp.exp(-0.5 * ((x - y) / self.lengthscale) ** 2)


# %% [markdown]
# Here, the Python `dataclass` is a class that simplifies the process of
# creating classes that primarily store data. It reduces boilerplate code and
# provides convenient methods for initialising and representing the data. An
# equivalent class could be written as:
#
# ```python
# class RBF:
#
#     def __init__(self, lengthscale: float = 1.0, variance: float = 1.0) -> None:
#         self.lengthscale = lengthscale
#         self.variance = variance
#
#     def covariance(self, x: jax.Array, y: jax.Array) -> jax.Array:
#         return self.variance * jnp.exp(-0.5 * (jnp.linalg.norm(x, y) / self.lengthscale)**2)
# ```

# %% [markdown]
# To establish some terminology, within the above RBF `dataclass`, we refer to
# the lengthscale and variance as _fields_. Further, the `RBF.covariance` is a
# _method_. So far so good. However, if e.g., we wanted to take the gradient of
# the kernel with repsect to its parameters $\nabla_{\ell, \sigma^2} k(1.0, 2.0;
# \ell, \sigma^2)$ at inputs $x=1.0$ and $y=2.0$:

# %%
kernel = RBF()

try:
    jax.grad(lambda kern: kern.covariance(1.0, 2.0))(kernel)
except TypeError as e:
    print(e)

# %% [markdown]
# We get an error. This is since, the object we have defined is not yet
# compatible with JAX. To achieve this we must consider JAX's _PyTree_
# abstraction.

# %% [markdown]
# ## PyTree’s
#
# JAX pytrees are a powerful tool in the JAX library that enables you to work
# with complex data structures in a way that is efficient, flexible, and easy to
# use. A PyTree is simply a data structure that is composed of other data
# structures, and it can be thought of as a tree where each 'node' is either a
# leaf (a simple data structure) or another PyTree. By default, the default set
# of 'node' types that are regarded a PyTree are Python lists, tuples, and
# dicts.
#
# For instance,

# %%
tree = [3.14, {"Monte": object(), "Carlo": False}]
print(tree)

# %% [markdown]
# is a PyTree with structure

# %%
import jax.tree_util as jtu

print(jtu.tree_structure(tree))

# %% [markdown]
# with the following leaves

# %%
print(jtu.tree_leaves(tree))

# %% [markdown]
# Consider a second example, a _PyTree of JAX arrays_

# %%
tree = (
    jnp.array([1.0, 2.0, 3.0]),
    jnp.array([4.0, 5.0, 6.0]),
    jnp.array([7.0, 8.0, 9.0]),
)

# %% [markdown]
# You can use this template to perform various operations on the data, such as
# applying a function to each leaf of the PyTree.
#
#
#
# For example, suppose you want to square each element of the arrays. You can
# then apply this using the `tree_map` function from the `jax.tree_util` module:

# %%
print(jtu.tree_map(lambda x: x**2, tree))


# %% [markdown]
# In this example, the PyTree makes it easy to apply a function to each leaf of
# a complex data structure, without having to manually traverse the data
# structure and handle each leaf individually. JAX PyTrees, therefore, are a
# powerful tool that can simplify many tasks in machine learning and scientific
# computing. As such, most JAX functions operate over _PyTrees of JAX arrays_.
# For instance, `jax.lax.scan`, accepts as input and produces as output a
# pytrees of JAX arrays.
#
# Another key advantages of using JAX PyTrees is that they are designed to work
# efficiently with JAX's automatic differentiation and compilation features. For
# example, suppose you have a function that takes a PyTree as input and returns
# a scalar value:


# %%
def sum_squares(x):
    return jnp.sum(x[0] ** 2 + x[1] ** 2 + x[2] ** 2)


sum_squares(tree)

# %% [markdown]
# You can use JAX's `grad` function to automatically compute the gradient of
# this function with respect to the input pytree:

# %%
gradient = jax.grad(sum_squares)(tree)
print(gradient)

# %% [markdown]
# This computes the gradient of the `sum_squares` function with respect to the
# input pytree, and returns a new pytree with the same shape and structure.

# %% JAX PyTrees are also designed to be highly extensible, where [markdown]
# custom types can be readily registered through a global registry with the
# values of such traversed recursively (i.e., as a tree!). This means we can
# define our own custom data structures and use them as PyTrees. This is the
# functionality that we exploit, whereby we construct all Gaussian process
# models via a tree-structure through our `Module` object.

# %% [markdown]
# # Module
#
# Our design, first and foremost, minimises additional abstractions on top of
# standard JAX: everything is just PyTrees and transformations on PyTrees, and
# secondly, provides full compatibility with the main JAX library itself,
# enhancing integrability with the broader ecosystem of third-party JAX
# libraries. To achieve this, our core idea is represent all model objects via
# an immutable PyTree. Here the leaves of the PyTree represent the parameters
# that are to be trained, and we describe their domain and trainable status as
# `dataclass` metadata.
#
# For our RBF kernel we have two parameters; the lengthscale and the variance.
# Both of these have positive domains, and by default we want to train both of
# these parameters. To encode this we use a `param_field`, where we can define
# the domain of both parameters via a `Softplus` bijector (that restricts them
# to the positive domain), and define their trainble status to `True`.

# %%
import tensorflow_probability.substrates.jax.bijectors as tfb
from gpjax.base import Module, param_field


@dataclass
class RBF(Module):
    lengthscale: float = param_field(1.0, bijector=tfb.Softplus(), trainable=True)
    variance: float = param_field(1.0, bijector=tfb.Softplus(), trainable=True)

    def covariance(self, x: jax.Array, y: jax.Array) -> jax.Array:
        return self.variance * jnp.exp(-0.5 * ((x - y) / self.lengthscale) ** 2)


# %% [markdown]
# Here `param_field` is just a special type of `dataclasses.field`. As such the
# following:
#
# ```python
# param_field(1.0, bijector= tfb.Identity(), trainable=False)
# ```
#
# is equivalent to the following `dataclasses.field`
#
# ```python
# field(default=1.0, metadata={"trainable": False, "bijector": tfb.Identity()})
# ```
#
#
# By default unmarked leaf attributes default to an `Identity` bijector and
# trainablility set to `True`.

# %% [markdown]
#
# ### Replacing values
# For consistency with JAX’s functional programming principles, `Module`
# instances are immutable. PyTree nodes can be changed out-of-place via the
# `replace` method.

# %%
kernel = RBF()
kernel = kernel.replace(lengthscale=3.14)  # Update e.g., the lengthscale.
print(kernel)

# %% [markdown]
# ## Transformations 🤖
#
# Use `constrain` / `unconstrain` to return a `Module` with each parameter's
# bijector `forward` / `inverse` operation applied!

# %%
# Transform kernel to unconstrained space
unconstrained_kernel = kernel.unconstrain()
print(unconstrained_kernel)

# Transform kernel back to constrained space
kernel = unconstrained_kernel.constrain()
print(kernel)

# %% [markdown]
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
# Recall the example earlier, where we wanted to take the gradient of the kernel
# with repsect to its parameters $\nabla_{\ell, \sigma^2} k(1.0, 2.0; \ell,
# \sigma^2)$ at inputs $x=1.0$ and $y=2.0$. We can now confirm we can do this
# with the new `Module`.
#
#

# %%
kernel = RBF()

jax.grad(lambda kern: kern.covariance(1.0, 2.0))(kernel)

# %% [markdown]
# During gradient learning of models, it can sometimes be useful to fix certain
# parameters during the optimisation routine. For this, JAX provides a
# `stop_gradient` operand to prevent the flow of gradients during forward or
# reverse-mode automatic differentiation, as illustrated below for a function
# $f(x) = x^2$.

# %%
from jax import lax


def f(x):
    x = lax.stop_gradient(x)
    return x**2


jax.grad(f)(1.0)

# %% [markdown]
# We see that gradient return is `0.0` instead of `2.0` due to the stopping of
# the gradient. Analogous to this, we provide this functionality to gradient
# flows on our `Module` class, via a `stop_gradient` method.
#
# Setting a (leaf) parameter's trainability to false can be achieved via the
# `replace_trainable` method.

# %%

kernel = RBF()
kernel = kernel.replace_trainable(lengthscale=False)

jax.grad(lambda kern: kern.stop_gradient().covariance(1.0, 2.0))(kernel)

# %% [markdown]
# As expected, the gradient is zero for the lengthscale parameter.

# %% [markdown]
# ## Static fields

# %% [markdown]
# In machine learning, initialising model parameters from random points is a
# common practice because it helps to break the symmetry in the model and allows
# the optimization algorithm to explore different regions of the parameter
# space.
#
# We could cleanly do this within the RBF class via a `post_init` method as
# follows:

# %%
import jax.random as jr
import tensorflow_probability.substrates.jax.distributions as tfd
from dataclasses import field


@dataclass
class RBF(Module):
    lengthscale: float = param_field(
        init=False, bijector=tfb.Softplus(), trainable=True
    )
    variance: float = param_field(init=False, bijector=tfb.Softplus(), trainable=True)
    key: jr.KeyArray = field(default_factory=lambda: jr.PRNGKey(42))
    # Note, for Python <3.11, you can use the following line:
    # key: jr.KeyArray = jr.PRNGKey(42)

    def __post_init__(self):
        # Split key into two keys
        key1, key2 = jr.split(self.key)

        # Sample from Gamma distribution to initialise lengthscale and variance
        self.lengthscale = tfd.Gamma(1.0, 0.1).sample(seed=key1)
        self.variance = tfd.Gamma(1.0, 0.1).sample(seed=key2)

    def covariance(self, x: jax.Array, y: jax.Array) -> jax.Array:
        return self.variance * jnp.exp(-0.5 * ((x - y) / self.lengthscale) ** 2)


kernel = RBF()
print(kernel)

# %% [markdown]
# So far so good. But however, if we now took our gradient again

# %%
try:
    jax.grad(lambda kern: kern.stop_gradient().covariance(1.0, 2.0))(kernel)
except TypeError as e:
    print(e)

# %% [markdown]
# We observe that we get a TypeError because the key is not differentiable. We
# can fix this by using a `static_field` for defining our key attribute.

# %%
from gpjax.base import static_field

# from simple_pytree import static_field  # <- need to rebase and replace with above line.


@dataclass
class RBF(Module):
    lengthscale: float = param_field(
        init=False, bijector=tfb.Softplus(), trainable=True
    )
    variance: float = param_field(init=False, bijector=tfb.Softplus(), trainable=True)
    key: jr.KeyArray = static_field(jr.PRNGKey(42))

    def __post_init__(self):
        # Split key into two keys
        key1, key2 = jr.split(self.key)

        # Sample from Gamma distribution to initialise lengthscale and variance
        self.lengthscale = tfd.Gamma(1.0, 0.1).sample(seed=key1)
        self.variance = tfd.Gamma(1.0, 0.1).sample(seed=key2)

    def covariance(self, x: jax.Array, y: jax.Array) -> jax.Array:
        return self.variance * jnp.exp(-0.5 * ((x - y) / self.lengthscale) ** 2)


fixed_kernel = RBF()
print(fixed_kernel)

# %% [markdown]
# So we get the same class as before. But this time

# %%
jax.grad(lambda kern: kern.stop_gradient().covariance(1.0, 2.0))(fixed_kernel)

# %% [markdown]
# What happened to get the result we wanted? The difference lies in the
# treatment of the key attribute as a PyTree leaf in the first example, which
# caused the gradient computation to fail. Examining the flattened PyTree's of
# both cases:

# %%
print(jax.tree_util.tree_flatten(fixed_kernel))
print(jax.tree_util.tree_flatten(kernel))

# %% [markdown]
# We see that assigning `static_field` tells JAX not to regard the attribute as
# leaf of the PyTree.

# %% [markdown]
# ## Metadata

# %% [markdown]
# To determine the parameter domain and trainable statuses of each parameter,
# the `Module` stores metadata for each leaf of the PyTree. This metadata is
# defined through a `dataclasses.field`. Thus, under the hood, we can define our
# `RBF` kernel object (equivalent to before) manually as follows:

# %%
from dataclasses import field


@dataclass
class RBF(Module):
    lengthscale: float = field(
        default=1.0, metadata={"bijector": tfb.Softplus(), "trainable": True}
    )
    variance: float = field(
        default=1.0, metadata={"bijector": tfb.Softplus(), "trainable": True}
    )

    def covariance(self, x: jax.Array, y: jax.Array) -> jax.Array:
        return self.variance * jnp.exp(-0.5 * ((x - y) / self.lengthscale) ** 2)


# %% [markdown]
# Here the `metadata` in the `dataclasses.field`, defines the metadata we
# associate with each PyTree leaf. This metadata can be a dictionary of any
# attributes we wish to store about each leaf. For example, we could extend this
# further by introducing a `name` attribute:

# %%
from dataclasses import field


@dataclass
class RBF(Module):
    lengthscale: float = field(
        default=1.0,
        metadata={"bijector": tfb.Softplus(), "trainable": True, "name": "lengthscale"},
    )
    variance: float = field(
        default=1.0,
        metadata={"bijector": tfb.Softplus(), "trainable": True, "name": "variance"},
    )

    def covariance(self, x: jax.Array, y: jax.Array) -> jax.Array:
        return self.variance * jnp.exp(-0.5 * ((x - y) / self.lengthscale) ** 2)


# %% [markdown]
# We can trace the metadata defined on the class via `meta_leaves`.

# %%
from gpjax.base import meta_leaves

rbf = RBF()

meta_leaves(rbf)

# %% [markdown]
# Similar to `jax.tree_utils.tree_leaves`, this function returns a flattened
# PyTree. However, instead of just the values, it returns a list of tuples that
# contain both the metadata and value of each PyTree leaf. This traced metadata
# can be utilised for applying maps (how `constrain`, `unconstrain`,
# `stop_gradient` work), as described in the next section.

# %% [markdown]
# ## Metamap

# %% [markdown]
# The `constrain`, `unconstrain`, and `stop_gradient` methods on the `Module`
# use a `meta_map` function under the hood. This function enables us to apply
# metadata functions to the PyTree leaves, making it a powerful tool.
#
# To achieve this, the function involves the same tracing as `meta_leaves` to
# create a flattened list of tuples consisting of (metadata, leaf value).
# However, it also allows us to apply a function to this list and return a new
# transformed PyTree, as demonstrated in the examples that follow.

# %% [markdown]
# ### Filter example:

# %% A `meta_map` works similarly to `jax.tree_utils.tree_map`. [markdown]
# However, it differs in that it allows us to define a function that operates on
# the tuple (metadata, leaf value). For example, we could use a function to
# filter based on a `name` attribute.

# %%
from gpjax.base import meta_map


def filter_lengthscale(meta_leaf):
    meta, leaf = meta_leaf
    if meta.get("name", None) == "lengthscale":
        return 3.14
    else:
        return leaf


print(meta_map(filter_lengthscale, rbf))

# %% [markdown]
# ### How `constrain` works:

# %% [markdown]
# To apply a constrain, we filter on the attribute "bijector", and apply a
# forward transformation to the PyTree leaf:

# %%


# This is how constrain works.
def _apply_constrain(meta_leaf):
    meta, leaf = meta_leaf

    if meta is None:
        return leaf

    return meta.get("bijector", tfb.Identity()).forward(leaf)


meta_map(_apply_constrain, rbf)

# %% [markdown]
# As expected, we find the same result as calling `rbf.constrain()`.
