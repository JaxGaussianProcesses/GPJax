:hide-toc: true


Welcome to GPJax's documentation!
=================================

GPJax is a didactic Gaussian process that supports GPU acceleration and just-in-time compilation. The intention of GPJax is to provide a framework for researchers to rapidly prototype and develop new Gaussian process methods. We seek to provide this by providing an API that seeks to best represent the underlying mathematics of Gaussian processes.

You can view the source code for GPJax `here on Github <https://github.com/thomaspinder/GPJax>`_

Gaussian process 'Hello World' example
-----------------------------------------------------

To gain an intuition for the exposed API provided by GPJax, a simple example of that derives the Gaussian process posterior for regression can be represented by::

    import gpjax
    import jax.numpy as jnp
    import jax.random as jr
    key = jr.PRNGKey(123)

    kernel = gpjax.kernels.RBF()
    f_prior = gpjax.gps.Prior(kernel = kernel)

    likelihood = gpjax.likelihoods.Gaussian()

    posterior = f_prior * likelihood


.. toctree::
    :maxdepth: 1
    :caption: Getting Started

    installation
    design

.. toctree::
    :maxdepth: 1
    :caption: Examples

    nbs/regression
    nbs/classification
    nbs/sparse_regression
    nbs/graph_kernels
    nbs/barycentres

.. toctree::
    :maxdepth: 1
    :caption: Guides

    nbs/kernels

.. toctree::
    :maxdepth: 1
    :caption: Experimental

    nbs/tfp_integration

.. toctree::
    :maxdepth: 1
    :caption: Package Reference

    api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


Bibliography
-----------------------------

.. bibliography::
    :cited:
