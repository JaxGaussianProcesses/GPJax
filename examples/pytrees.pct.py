# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
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
# # Guide to PyTrees
# ## Immutable strucutres (Dan)
#
# - Intro to PyTrees
# - Why PyTrees
# - Diagrams labelling the components of a PyTree
#
# ## Operations on PyTrees (Tom)
#
# - Import the RBF kernel
#    - Computing gradients
#    - Squaring the leaves
#    - Adding two PyTrees
#
# ## Writing your own PyTree
#
# - Give an example of creating the `Constant` mean function
#     - Demonstrate fixing the mean vs. learning it
#     - Transforming the parameter's value
#     - Changing the bijection
