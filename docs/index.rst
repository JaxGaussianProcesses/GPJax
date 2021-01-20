Welcome to GPJax's documentation!
=================================

GPJax is a didactic Gaussian process with GPU acceleration and just-in-time compilation. The intention of GPJax is to provide a framework for researchers to rapidly prototype and develop new Gaussian process methods. We seek to provide this by providing an API that seeks to best represent the underlying mathematics of Gaussian processes.

Gaussian process 'Hello World' example
-----------------------------------------------------

To gain an intuition for the exposed API provided by GPJax, a simple example of that derives the Gaussian process posterior for regression can be represented by::

    import jax.numpy as jnp
    import jax.random as jr
    import gpjax
    key = jr.PRNGKey(123)

    X = jnp.linspace(-2., 2., 100)
    y = jnp.sin(x) + jr.normal(key, shape=X.shape)*0.05

    kernel = gpjax.kernels.RBF()
    f_prior = gpjax.gps.Prior(kernel)

    likelihood = gpjax.likelihoods.Gaussian()

    posterior = f_prior * likelihood

Contents
-----------------------------------------------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:


GP Priors
-----------------------------------------------------

.. toctree::
   :maxdepth: 2

   priors



Kernels
-----------------------------------------------------

.. toctree::
   :maxdepth: 2

   kernels

Parameters
-----------------------------------------------------

.. toctree::
   :maxdepth: 2

   parameters

