:hide-toc: true


Welcome to GPJax!
=================================

GPJax is a didactic Gaussian process library that supports GPU acceleration and just-in-time compilation. We seek to provide a flexible API as close as possible to how the underlying mathematics is written on paper to enable researchers to rapidly prototype and develop new ideas.

.. image:: ./GP.pdf
  :width: 800
  :alt: Alternative text

You can view the source code for GPJax `here on Github <https://github.com/thomaspinder/GPJax>`_.

'Hello World' example
-----------------------------------------------------

Defining a Gaussian process posterior is simple as typing the maths we would write on paper.

.. code-block:: python

    import gpjax as gpx

    kernel = gpx.kernels.RBF()
    prior = gpx.gps.Prior(kernel = kernel)

    likelihood = gpx.likelihoods.Gaussian(num_datapoints = 123)

    posterior = prior * likelihood

To learn more, checkout the `regression notebook <https://gpjax.readthedocs.io/en/latest/nbs/regression.html>`_.

.. toctree::
    :maxdepth: 1
    :caption: Getting Started
    :hidden:

    installation
    design

.. toctree::
    :maxdepth: 1
    :caption: Examples
    :hidden:

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
    :hidden:

    nbs/kernels
    nbs/yacht

.. toctree::
    :maxdepth: 1
    :caption: Package Reference
    :hidden:

    api

Indices and tables
-----------------------------
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


Bibliography
-----------------------------

.. bibliography::
    :cited:
