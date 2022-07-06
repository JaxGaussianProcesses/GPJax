:hide-toc: true


Welcome to GPJax's documentation!
=================================

GPJax is a didactic Gaussian process library that supports GPU acceleration and just-in-time compilation. We seek to provide a flexible API as close as possible to how the underlying mathematics is written on paper to enable researchers rapidly prototype and develop new ideas.

You can view the source code for GPJax `here on Github <https://github.com/thomaspinder/GPJax>`_.

Gaussian process 'Hello World' example
-----------------------------------------------------

Defining a Gaussian process posterior is simple as typing the maths we would write on paper.::

    import gpjax as gpx

    kernel = gpx.kernels.RBF()
    prior = gpx.gps.Prior(kernel = kernel)

    likelihood = gpx.likelihoods.Gaussian(num_datapoints = 1729)

    posterior = prior * likelihood

To learn more, checkout the `regression notebook <https://gpjax.readthedocs.io/en/latest/nbs/regression.html>`_.


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
    nbs/uncollapsed_vi
    nbs/collapsed_vi
    nbs/graph_kernels
    nbs/barycentres
    nbs/haiku
    nbs/tfp_integration

.. toctree::
    :maxdepth: 1
    :caption: Guides

    nbs/kernels

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
