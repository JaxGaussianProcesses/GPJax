:hide-toc: true


Welcome to GPJax's documentation!
=================================

GPJax is a didactic Gaussian process library that supports GPU acceleration and just-in-time compilation. We seek to provide a flexible API as close as possible to how the underlying mathematics is written on paper to enable researchers rapidly prototype and develop new ideas.

You can view the source code for GPJax `here on Github <https://github.com/thomaspinder/GPJax>`_.

Gaussian process 'Hello World' example
-----------------------------------------------------

For intuition of GPJax's exposed API, a simple example that derives the Gaussian process posterior for regression can be represented by::

    import gpjax
    import jax.numpy as jnp
    import jax.random as jr
    key = jr.PRNGKey(123)

    kernel = gpjax.kernels.RBF()
    prior = gpjax.gps.Prior(kernel = kernel)

    likelihood = gpjax.likelihoods.Gaussian()

    posterior = prior * likelihood


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
    nbs/haiku
    nbs/t_regression

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
