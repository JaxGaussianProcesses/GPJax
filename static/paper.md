---
title: "GPJax: A Gaussian Process Framework in JAX"
tags:
  - Python
  - Gaussian processes
  - JAX
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

Gaussian processes [GPs, @rasmussen2006gaussian] are Bayesian nonparametric models that have been successfully used in applications such as geostatistics [@matheron1963principles], Bayesian optimisation [@mockus1978application], and reinforcement learning [@deisenroth2011pilco]. `GPJax` is a didactic GP library targeted at researchers who wish to develop novel GP methodology. The scope of `GPJax` is to provide users with a set of composable objects for constructing GP models that closely resemble the underlying maths that one would write on paper. Furthermore, by the virtue of being written in JAX [@jax2018github], `GPJax` natively supports CPUs, GPUs and TPUs through efficient compilation to XLA, automatic differentiation and vectorised operations. Consequently, `GPJax` provides a modern GP package that can effortlessly be tailored, extended and interleaved with other libraries to meet the individual needs of researchers and scientists.

# Statement of Need

From both an applied and methodological perspective, GPs are widely employed in the statistics and machine learning communities. High-quality software packages that promote GP modelling are accountable for much of their success. However, there currently exists a gap within the JAX ecosystem for a Gaussian process package to be developed that incorporates scalable inference techniques. `GPJax` seeks to resolve this.

`GPJax` has been carefully tailored to amalgamate with the JAX ecosystem. For efficient Markov Chain Monte Carlo inference, `GPJax` can utilise samplers from BlackJax [@blackjax2021] and TensorFlow Probability [@abadi2016tensorflow]. For gradient-based optimisation, `GPJax` integrates seamlessly with Optax [@deepmind2020jax], providing a vast suite of optimisers and learning rate schedules. To efficiently represent probability distributions, `GPJax` leverages Distrax [@deepmind2020jax] and TensorFlow Probability [@abadi2016tensorflow]. To combine GPs with deep learning methods, `GPJax` can incorporate the functionality provided within Haiku [@deepmind2020jax]. The `GPJax` documentation includes examples of each of these integrations.

The foundation of each abstraction given in `GPJax` is a Chex [@deepmind2020jax] dataclass object. These require significantly less boilerplate code than regular Python classes, leading to a more readable codebase. Moreover, Chex dataclasses are registered as PyTree nodes, facilitating the applications of JAX operations such as just-in-time compilation and automatic differentiation to any `GPJax` object.

The intimacy between `GPJax` and the underlying maths also makes `GPJax` an excellent package for people new to GP modelling. Having the ability to easily cross-reference the contents of a textbook with the code that one is writing is invaluable when trying to build an intuition for a new statistical method. We further support this effort in `GPJax` through documentation that provides detailed explanations of the operations conducted within each notebook.

# Wider Software Ecosystem

Within the Python community, the three most popular packages for GP modelling are GPFlow [@matthews2017gpflow], GPyTorch [@gardner2018gpytorch], and GPy [@gpy2014]. Despite these packages being indispensable tools for the community, none support integration with a JAX-based workflow. On the other hand, BayesNewton [@wilkinson2021bayesnewton] and TinyGP [@dfm2021tinygp] packages utilise a Jax backend. However, BayesNewton is designed on top of ObJax [@objax2020github], making integration with the broader Jax ecosystem challenging. Meanwhile, TinyGP offers excellent integration with inference frameworks such as NumPyro [@phan2019composable] but does not yet support inducing points frameworks [e.g., @hensman2013gaussian]. `GPJax` exists to resolve these issues. Furthermore, modern research from the GP literature, graph kernels [@borovitskiy2021matern] and Wasserstein barycentres for GPs [@mallasto2017learning], for example, are supported within `GPJax` but absent from these packages. Finally, the Stheno package [@stheno2022bruinsma] supports a JAX backend along with TensorFlow, PyTorch and Numpy. Whilst this integrates GPs into an extensive JAX workflow, `GPJax` has the advantage of being a pure JAX codebase, whereas Stheno requires using a custom linear algebra framework.

For completeness, packages written for languages other than Python include GPML [@rasmussen2010gaussian] and GPStuff [@vanhatalo2013gpstuff] in MATLAB. An R port also exists for GPStuff. Within Julia, there exists GaussianProcesses.jl [@fairbrother2022gaussianprocesses], AugmentedGaussianProcesses.jl [@fajou20a] and Stheno.jl [@stheno2022tebbutt].

GP implementations are available in numerous modern probabilistic programming languages such as NumPyro [@phan2019composable], Stan [@carpenter2017stan], and PyMC [@Salvatier2016].

# External Usage

Two recent research papers [@pinder2021gaussian; @pinder2022street] utilise the graph kernel functionality provided by `GPJax`. Furthermore, `GPJax` is being used to build probabilistic ensembles of climate models [@amos2022ensembles] and perform adaptive sampling in deep-sea environmental [@dodd2022ensembles].

# Acknowledgments

As an open-source project, `GPJax` has benefitted from contributions made by the wider community. We especially thank Juan Emmanuel Johnson and are grateful for the thoughts and advice from the wider GP community.

# Funding Statement

TP is supported by the Data Science for the Natural Environment project (EPSRC grant number EP/R01860X/1). DD is supported by the STOR-i Centre for Doctoral Training (EPSRC grant number EP/S022252/1) and the Research Hub for Transforming Energy Infrastructure through Digital Engineering (ARC grant number IH200100009).

# References
