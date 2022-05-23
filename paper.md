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

Gaussian processes (GPs) are a widely used model in the machine learning and statistics community [@rasmussen2006gaussian]. In `GPJax`, we seek to build computational abstractions of GPs such that the code closely, if not exactly, mimics the underlying maths that one would write on paper. The result of this is that `GPJax` provides a modern GP package that can easily be tailored and extended to meet the individual needs of researchers and scientists wishing to develop their own GP methodology.

`GPJax` is a didactic GP library written in Jax [@jax2018github] that is targetted at researchers who wish to develop novel GP methodology. The scope of `GPJax` is to provide users with a set of composable objects that can be combined to construct GP models. Code is written in a manner that is easy to extend and adapt to a user's own unique needs through a design that is easily interleaved with other libraries.

By writing `GPJax` using Jax, it is possible to run all `GPJax` code on CPUs, GPUs or TPUs through efficient compilation to XLA. In addition to this, automatic differentiation and vectorised operations are natively supported in `GPJax` through its Jax underpinning.

# Statement of Need

The design of `GPJax` has been carefully tailored to enable easy integration with other packages in the Jax ecosystem. Currently, Markov Chain Monte Carlo can be conducted using either BlackJax [@blackjax2021] or TensorFlow [@abadi2016tensorflow]. Seamless integration with Optax [@deepmind2020jax] provides a suite of gradient based optimisers and learning rate schedulers. `GPJax` leverages Distrax [@deepmind2020jax] and TensorFlow Probability [@abadi2016tensorflow] to efficiently represent probability distributions and their corresponding functions. To combine GPs with deep learning methods, `GPJax` integrates with the functionality provided within Haiku [@deepmind2020jax]. Each of these integrations is documented with example usage in the `GPJax` documentation.

The foundation of each abstraction given in `GPJax` is a Chex [deepmind2020jax] dataclass object. These are well-suited to `GPJax` as they require significantly less boilerplate code than regular Python classes, thus leading to a more readable codebase. Unlike regular Python dataclasses though, Chex dataclasses are internally registered as PyTree nodes, thus enabling Jax operations such as just-in-time compilation and automatic differentiate to be applied to any `GPJax` object.

The intimacy between `GPJax` and the underlying maths also makes `GPJax` an excellent package for people new to GP modelling. Having the ability to easily cross-reference the contents of a textbook with the code that one is writing is an invaluable asset when one is trying to first build an intuition for a new statistical method. We further support this effort in `GPJax` through documentation that provides detailed explanations of the operations being conducted within each notebook.

# Wider Software Ecosystem

From both an applied and methodological perspective, GP models are a widely used tool in the statistics and machine learning community. Much of this success can be attributed to range of high-quality software packages that faciliate GP modelling. Within the Python community, the three most popular packages for GP modelling are GPFlow [@matthews2017gpflow], GPyTorch [@gardner2018gpytorch], and GPy [@gpy2014]. Despite each of these packages being indispensible tools for the community, none of them support integration with a Jax-based workflow. `GPJax` seeks to resolve this issue. Furthermore, modern research from the GP literature such as graph kernels [@borovitskiy2021matern] and Wasserstein barycentres for GPs [@mallasto2017learning] are supported within GPJax but absent from all other packages.

For completeness, packages written for languages other than Python include GPML [@rasmussen2010gaussian] in Matlab, GaussianProcesses.jl [@fairbrother2022gaussianprocesses], AugmentedGaussianProcesses.jl [@fajou20a] and Stheno [@stheno2022tebbutt] all in Julia.

# External usage

The graph kernel functionality supported by `GPJax` has been used within two recent research papers: [@pinder2021gaussian] and [@pinder2022street].

# Funding Statement

TP is supported by the Data Science for the Natural Environment project (EPSRC grant number EP/R01860X/1). DD is supported by the EPSRC funded STOR-i centre for doctoral training (EP/S022252/1) and the ARC Research Hub for Transforming Energy Infrastructure through Digital Engineering (IH200100009).

# References
