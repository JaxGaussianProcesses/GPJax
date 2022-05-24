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

date: 24 May 2022
bibliography: paper.bib
---

# Summary

Gaussian processes (GPs) [@rasmussen2006gaussian] are Bayesian nonparametric models that have been successfully used in applications such as geostatistics [@matheron1963principles], Bayesian optimisation [@mockus1978application], and reinforcement learning [@deisenroth2011pilco]. In `GPJax`, we seek to build computational abstractions of GPs that closely resemble the underlying maths that one would write on paper. Consequently, `GPJax` provides a modern GP package that can easily be tailored and extended to meet the individual needs of researchers and scientists wishing to develop their GP methodology.

`GPJax` is a didactic GP library targeted at researchers who wish to develop novel GP methodology. The scope of `GPJax` is to provide users with a set of composable objects for constructing GP models, written in a manner that is easy to extend and adapt to a user's own unique needs through an interleaved design with other libraries.

`GPJax` is written in Jax [@jax2018github] and it is possible to run all `GPJax` code on CPUs, GPUs or TPUs through efficient compilation to XLA. In addition to this, `GPJax` natively supports automatic differentiation and vectorised operations through its Jax underpinning.

# Statement of Need

`GPJax` has been carefully tailored to amalgamate with the Jax ecosystem. For efficient Markov Chain Monte Carlo inference, `GPJax` can utilise samplers from BlackJax [@blackjax2021] and TensorFlow Probability [@abadi2016tensorflow]. For gradient-based optimisation, `GPJax` integrates seamlessly with Optax [@deepmind2020jax], providing a vast suite of optimisers and learning rate schedules. To efficiently represent probability distributions, `GPJax` leverages Distrax [@deepmind2020jax] and TensorFlow Probability [@abadi2016tensorflow]. To combine GPs with deep learning methods, `GPJax` can incorporate the functionality provided within Haiku [@deepmind2020jax]. The `GPJax` documentation includes examples of each of these integrations.

The foundation of each abstraction given in `GPJax` is a Chex [deepmind2020jax] dataclass object. These require significantly less boilerplate code than regular Python classes, leading to a more readable codebase. Moreover, Chex dataclasses are registered as PyTree nodes, facilitating the applications of Jax operations such as just-in-time compilation and automatic differentiation to any `GPJax` object.

The intimacy between `GPJax` and the underlying maths also makes `GPJax` an excellent package for people new to GP modelling. Having the ability to easily cross-reference the contents of a textbook with the code that one is writing is invaluable when trying to build an intuition for a new statistical method. We further support this effort in `GPJax` through documentation that provides detailed explanations of the operations conducted within each notebook.

# Wider Software Ecosystem

From both an applied and methodological perspective, GPs are widely employed in the statistics and machine learning communities. High-quality software packages that promote GP modelling are accountable for much of their success. Within the Python community, the three most popular packages for GP modelling are GPFlow [@matthews2017gpflow], GPyTorch [@gardner2018gpytorch], and GPy [@gpy2014]. Despite each of these packages being indispensable tools for the community, none of them supports integration with a Jax-based workflow. `GPJax` seeks to resolve this issue. Furthermore, modern research from the GP literature, graph kernels [@borovitskiy2021matern] and Wasserstein barycentres for GPs [@mallasto2017learning], for example, are supported within `GPJax` but absent from all other packages.

For completeness, packages written for languages other than Python include GPML [@rasmussen2010gaussian] in Matlab, GaussianProcesses.jl [@fairbrother2022gaussianprocesses], AugmentedGaussianProcesses.jl [@fajou20a] and Stheno [@stheno2022tebbutt] all in Julia.

# External usage

Two recent research papers [@pinder2021gaussian] and [@pinder2022street] use the graph kernel functionality provided in `GPJax`.

# Funding Statement

TP is supported by the Data Science for the Natural Environment project (EPSRC grant number EP/R01860X/1). DD is supported by the EPSRC funded STOR-i Centre for Doctoral Training (EP/S022252/1) and the ARC Research Hub for Transforming Energy Infrastructure through Digital Engineering (IH200100009).

# References
