<!-- <h1 align='center'>GPJax</h1>
<h2 align='center'>Gaussian processes in Jax.</h2> -->
<p align="center">
<img width="700" height="300" src="https://raw.githubusercontent.com/JaxGaussianProcesses/GPJax/main/docs/static/gpjax_logo.svg" alt="GPJax's logo">
</p>

[![codecov](https://codecov.io/gh/JaxGaussianProcesses/GPJax/branch/master/graph/badge.svg?token=DM1DRDASU2)](https://codecov.io/gh/JaxGaussianProcesses/GPJax)
[![CodeFactor](https://www.codefactor.io/repository/github/jaxgaussianprocesses/gpjax/badge)](https://www.codefactor.io/repository/github/jaxgaussianprocesses/gpjax)
[![Netlify Status](https://api.netlify.com/api/v1/badges/d3950e6f-321f-4508-9e52-426b5dae2715/deploy-status)](https://app.netlify.com/sites/endearing-crepe-c2d5fe/deploys)
[![PyPI version](https://badge.fury.io/py/GPJax.svg)](https://badge.fury.io/py/GPJax)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04455/status.svg)](https://doi.org/10.21105/joss.04455)
[![Downloads](https://pepy.tech/badge/gpjax)](https://pepy.tech/project/gpjax)
[![Slack Invite](https://img.shields.io/badge/Slack_Invite--blue?style=social&logo=slack)](https://join.slack.com/t/gpjax/shared_invite/zt-1da57pmjn-rdBCVg9kApirEEn2E5Q2Zw)

[**Quickstart**](#simple-example)
| [**Install guide**](#installation)
| [**Documentation**](https://docs.jaxgaussianprocesses.com/)
| [**Slack Community**](https://join.slack.com/t/gpjax/shared_invite/zt-1da57pmjn-rdBCVg9kApirEEn2E5Q2Zw)

GPJax aims to provide a low-level interface to Gaussian process (GP) models in
[Jax](https://github.com/google/jax), structured to give researchers maximum
flexibility in extending the code to suit their own needs. The idea is that the
code should be as close as possible to the maths we write on paper when working
with GP models.

# Package organisation

## Contributions

We would be delighted to receive contributions from interested individuals and
groups. To learn how you can get involved, please read our [guide for
contributing](https://github.com/JaxGaussianProcesses/GPJax/blob/main/docs/contributing.md).
If you have any questions, we encourage you to [open an
issue](https://github.com/JaxGaussianProcesses/GPJax/issues/new/choose). For
broader conversations, such as best GP fitting practices or questions about the
mathematics of GPs, we invite you to [open a
discussion](https://github.com/JaxGaussianProcesses/GPJax/discussions).

Another way you can contribute to GPJax is through [issue
triaging](https://www.codetriage.com/what).  This can include reproducing bug reports,
asking for vital information such as version numbers and reproduction instructions, or
identifying stale issues. If you would like to begin triaging issues, an easy way to get
started is to
[subscribe to GPJax on CodeTriage](https://www.codetriage.com/jaxgaussianprocesses/gpjax).

As a contributor to GPJax, you are expected to abide by our [code of
conduct](docs/CODE_OF_CONDUCT.md). If you feel that you have either experienced or
witnessed behaviour that violates this standard, then we ask that you report any such
behaviours through [this form](https://jaxgaussianprocesses.com/contact/) or reach out to
one of the project's [_gardeners_](https://docs.jaxgaussianprocesses.com/GOVERNANCE/#roles).

Feel free to join our [Slack
Channel](https://join.slack.com/t/gpjax/shared_invite/zt-1da57pmjn-rdBCVg9kApirEEn2E5Q2Zw),
where we can discuss the development of GPJax and broader support for Gaussian
process modelling.


## Governance

GPJax was founded by [Thomas Pinder](https://github.com/thomaspinder). Today, the
project's gardeners are [daniel-dodd@](https://github.com/daniel-dodd),
[henrymoss@](https://github.com/henrymoss), [st--@](https://github.com/st--), and
[thomaspinder@](https://github.com/thomaspinder), listed in alphabetical order. The full
governance structure of GPJax is detailed [here](docs/GOVERNANCE.md). We appreciate all
[the contributors to
GPJax](https://github.com/JaxGaussianProcesses/GPJax/graphs/contributors) who have
helped to shape GPJax into the package it is today.


# Supported methods and interfaces

## Notebook examples

> - [**Conjugate Inference**](https://docs.jaxgaussianprocesses.com/_examples/regression/)
> - [**Classification**](https://docs.jaxgaussianprocesses.com/_examples/classification/)
> - [**Sparse Variational Inference**](https://docs.jaxgaussianprocesses.com/_examples/collapsed_vi/)
> - [**Stochastic Variational Inference**](https://docs.jaxgaussianprocesses.com/_examples/uncollapsed_vi/)
> - [**Laplace Approximation**](https://docs.jaxgaussianprocesses.com/_examples/classification/#laplace-approximation)
> - [**Inference on Non-Euclidean Spaces**](https://docs.jaxgaussianprocesses.com/_examples/constructing_new_kernels/#custom-kernel)
> - [**Inference on Graphs**](https://docs.jaxgaussianprocesses.com/_examples/graph_kernels/)
> - [**Pathwise Sampling**](https://docs.jaxgaussianprocesses.com/_examples/spatial/)
> - [**Learning Gaussian Process Barycentres**](https://docs.jaxgaussianprocesses.com/_examples/barycentres/)
> - [**Deep Kernel Regression**](https://docs.jaxgaussianprocesses.com/_examples/deep_kernels/)
> - [**Poisson Regression**](https://docs.jaxgaussianprocesses.com/_examples/poisson/)
> - [**Bayesian Optimisation**](https://docs.jaxgaussianprocesses.com/_examples/bayesian_optimisation/)

## Guides for customisation
>
> - [**Custom kernels**](https://docs.jaxgaussianprocesses.com/_examples/constructing_new_kernels/#custom-kernel)
> - [**UCI regression**](https://docs.jaxgaussianprocesses.com/_examples/yacht/)

## Conversion between `.ipynb` and `.py`
Above examples are stored in [examples](docs/examples) directory in the double
percent (`py:percent`) format. Checkout [jupytext
using-cli](https://jupytext.readthedocs.io/en/latest/using-cli.html) for more
info.

* To convert `example.py` to `example.ipynb`, run:

```bash
jupytext --to notebook example.py
```

* To convert `example.ipynb` to `example.py`, run:

```bash
jupytext --to py:percent example.ipynb
```

# Installation

## Stable version

The latest stable version of GPJax can be installed via
pip:

```bash
pip install gpjax
```

> **Note**
>
> We recommend you check your installation version:
> ```python
> python -c 'import gpjax; print(gpjax.__version__)'
> ```



## Development version
> **Warning**
>
> This version is possibly unstable and may contain bugs.

> **Note**
>
> We advise you create virtual environment before installing:
> ```
> conda create -n gpjax_experimental python=3.10.0
> conda activate gpjax_experimental
>  ```


Clone a copy of the repository to your local machine and run the setup
configuration in development mode.
```bash
git clone https://github.com/JaxGaussianProcesses/GPJax.git
cd GPJax
hatch env create
hatch shell
```

> We recommend you check your installation passes the supplied unit tests:
>
> ```python
> hatch run dev:test
> ```

# Citing GPJax

If you use GPJax in your research, please cite our [JOSS paper](https://joss.theoj.org/papers/10.21105/joss.04455#).

```
@article{Pinder2022,
  doi = {10.21105/joss.04455},
  url = {https://doi.org/10.21105/joss.04455},
  year = {2022},
  publisher = {The Open Journal},
  volume = {7},
  number = {75},
  pages = {4455},
  author = {Thomas Pinder and Daniel Dodd},
  title = {GPJax: A Gaussian Process Framework in JAX},
  journal = {Journal of Open Source Software}
}
```
