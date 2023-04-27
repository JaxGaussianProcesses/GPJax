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
# ---

# %% [markdown]
# # 🌳 GPJax PyTrees
#
# `GPJax` **represents all mathematical objects as PyTrees.** As a result
#
# - We have a simple API with a **TensorFlow / PyTorch feel** …
# - … that is **fully compatible** with JAX's functional paradigm.
#
# The core idea of our abstraction builds on the _Equinox_ library, where we
# seek to provide the Bayesian/GP analogue to their neural network focused
# abstractions. However, we extend their abstraction further to permit users to
# write standard Python classes, and provide to readily define (and alter)
# domains and training statuses of parameter for optimisation, contained as a
# single model object, that is fully compatible with JAX autogradients out of
# the box - no filtering required, and provide functionality for applying
# dataclass metadata to Pytree’s.
#

# %% [markdown]
# ## The need for custom objects
#
# Within this notebook, we'll be using the squared exponential, or RBF, kernel
# for illustratory purposes. For a pair of vectors $x, y \in \mathbb{R}^d$, its
# form can be mathematically given by
# $$ k(x, y) = \sigma^2\exp\left(\frac{\lVert x-y\rVert_{2}^2}{2\ell^2} \right) $$
# where $\sigma\in\mathbb{R}_{>0}$ is a variance parameter and
# $\ell\in\mathbb{R}_{>0}$ a lengthscale parameter. We call the evaluation of
# $k(x, y)$ the _covariance_.
#
# Using a `dataclass`, we can represent this object as follows
# ```python
# from jax import Array
# from dataclasses import dataclass
#
# @dataclass
# class RBF:
# 	lengthscale: float
# 	variance: float
#
# 	def covariance(self, x: Array, y: Array) -> Array:
# 		pass
# ```
# We have for now left `covariance` empty; however, through this notebook, we shall
# build up to a fully object that can compute covariances in a JAX-compatible way.
#
# For those users who have not seen a `dataclass` before, this statement is equivalent
# to writing
# ```python
# class RBF:
# 	def __init__(self, lengthscale: float, variance: float) -> None:
# 		self.lengthscale = lengthscale
# 		self.variance = variance
#
# 	def covariance(self, x: Array, y: Array) -> Array:
# 		pass
# ```
# However, it a dataclass allows us to significantly reduce the number of lines
# of code needed to represent such objects, particularly as the code's
# complexity increases, as we shall go on to see.
#
# To establish some terminology, within the above RBF `dataclass`, we refer to
# the lengthscale and variance as _fields_. Further, the `RBF.covariance()` is a
# _method_.
#
# # A primer on PyTree’s:
#
# To efficiently represent data JAX provides a `*PyTree*` abstraction. PyTree’s as such, are immutable tree-like structure built out of *‘node’ types* —— container-like Python objects. For instance,
#
# ```python
# [3.14, {"Monte": object(), "Carlo": False}]
# ```
#
# is a PyTree with structure `[*, {"Monte": *, "Carlo": *}]` and leaves `3.14`, `object()`, `False`. As such, most JAX functions operate over pytrees, e.g., `jax.lax.scan`, accepts as input and produces as output a pytrees of JAX arrays.
#
# While the default set of ‘node’ types that are regarded internal pytree nodes is limited to objects such as lists, tuples, and dicts, JAX permits custom types to be readily registered through a global registry, with the values of such traversed recursively (i.e., as a tree!). This is the functionality that we exploit, whereby we construct all Gaussian process models via a tree-structure.
#
# - RBF
# - (1) In GPJax, RBF we inherit from abstract kernel.
# - (2) We don’t want to do this. We will do it from Module.
# - (3) Do it like the old code, as a property → kernel.gram() → separate function. covariance.
#
# # Module
#
# Our design, first and foremost, minimises additional abstractions on top of standard JAX: everything is just PyTrees and transformations on PyTrees, and secondly, provides full compatibility with the main JAX library itself, enhancing integrability with the broader ecosystem of third-party JAX libraries. To achieve this, our core idea is represent all model objects via an immutable tree-structure.
#
# Lets do an **RBF** kernel.
#
# ### Objects as immutable class instances:
#
# Equinox represents parameterised functions as class instances. These instances are immutable, for consistency with JAX’s functional programming principles. (And parameters updates occur out-of- place.)
#
# *Lets do an **RBF** kernel.*
#
# ```python
#
# ```
#
# ### Parameters and their metadata:
#
# We extend on Equinox’s functionality by consider the definition of parameters.
#
# There are two main considerations for model parameters, their:
#
# - Trainability status.
# - Domain.
#
# The former
#
#
#
# Equinox represents parameterised functions as class instances. These instances are immutable, for consistency with JAX’s functional programming principles. (And parameters updates occur out-of- place.)
#
# Why normalising flows don’t break the convention.
#
# **Our functional view:**
#
# - *We view all models as immutable trees.*
# - *What is computation if you have it on a model?*

# %% [markdown]
#
