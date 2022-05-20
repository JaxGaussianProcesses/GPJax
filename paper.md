---
title: "GPJax: A Gaussian Process Framework in Jax"
tags:
  - Python
  - Gaussian processes
  - Jax
  - Bayesian inference
  - Machine learning
  - Statistics
authors:
  - name: Thomas Pinder^[corresponding author]
    affiliation: "1"
  - name: Daniel Dodd
    affiliation: "2"
affiliations:
  - name: Department of Mathematics and Statistics, Lancaster University, United Kingdom
    index: 1
  - name: STOR-i Centre for Doctoral Training, Lancaster University, United Kingdom
    index: 2

date: 11 May 2022
bibliography: paper.bib
---

# Summary

GPJax is a didactic Gaussian process (GP) library written in Jax that is targetted at researchers who wish to develop novel GP methodology. The scope of GPJax is to provide users with the core objects used for constructing GP models in a composable manner. Code is purposefully written in a manner that is easy to extend and adapt to a user's own unique needs.

GPJax has been written using Jax [@jax2018github]. This design choice enables all code to be run on CPUs, GPUs or TPUs through efficient compilation to XLA. In addition to this, automatic differentiation and vectorised operations are natively supported in GPJax through its Jax underpinning.

# Statement of Need

In GPJax, we seek to build computational abstractions of GPs such that the code closely, if not exactly, mimics the underlying maths that one would write on paper.

# Wider Software Ecosystem

From both an applied and methodological perspective, GP models have been a widely used tool in the statistics and machine learning community. Much of this success can be attributed to range of high-quality software packages that faciliate GP modelling. Within the Python community, the three most popular packages for GP modelling are GPFlow [@matthews2017gpflow], GPyTorch [@gardner2018gpytorch], and GPy [@gpy2014]. Despite each of these packages being indispensible tools for the community, none of them support integration with a Jax-based workflow. GPJax seeks to resolve this issue.

For completeness, packages written for languages other than Python include GPML [@rasmussen2010gaussian] in Matlab, GaussianProcesses.jl [@fairbrother2022gaussianprocesses], AugmentedGaussianProcesses.jl [@fajou20a] and Stheno [@stheno2022tebbutt] all in Julia.

# External usage

The graph kernel functionality supported by GPJax has been used within two recent research papers: [@pinder2021gaussian] and [@pinder2022street].

# Funding Statement

TP is supported by the Data Science for the Natural Environment project (EPSRC grant number EP/R01860X/1).

# References
