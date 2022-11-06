.. role:: hidden
    :class: hidden-section

**********************
Package Reference
**********************

Gaussian Processes
#################################

The Gaussian process abstractions in GPJax can be segmented into two distinct
types: prior and posterior objects. This makes for a clean separation of both
code and mathematical concepts. Throught the multiplication of a
`Prior <https://gpjax.readthedocs.io/en/latest/api.html#gaussian-process-priors>`_
and `Likelihood <https://gpjax.readthedocs.io/en/latest/api.html#module-gpjax.likelihoods>`_,
GPJax will then return the appropriate `Posterior <https://gpjax.readthedocs.io/en/latest/api.html#gaussian-process-posteriors>`_.


.. automodule:: gpjax.gps
.. currentmodule:: gpjax.gps

Abstract GPs
*********************************

To ensure a consistent API, we subclass all Gaussian process objects from either
``AbstractPrior`` or ``AbstractPosterior``. These classes are not intended to be
used directly, but instead to provide a common interface for all downstream Gaussian
process objects.

.. autoclass:: AbstractPrior
   :members:
   :special-members: __call__
   :private-members: _initialise_params
   :exclude-members: from_tuple, replace, to_tuple

.. autoclass:: AbstractPosterior
   :members:
   :special-members: __call__


Gaussian Process Priors
*********************************

.. autoclass:: Prior
   :members:
   :special-members: __call__, __mul__

Gaussian Process Posteriors
*********************************

There are two main classes of posterior Gaussian process objects within GPJax.
The ``ConjugatePosterior`` class is used when the likelihood distribution is
Gaussian whilst the ``NonConjugatePosterior`` class is used when the likelihood
distribution is non-Gaussian.

.. autoclass:: ConjugatePosterior
   :members:
   :special-members: __call__

.. autoclass:: NonConjugatePosterior
   :members:
   :special-members: __call__

Posterior Constructors
*********************************

.. autofunction:: construct_posterior


Kernels
########################

.. automodule:: gpjax.kernels
.. currentmodule:: gpjax.kernels

Kernel Functions
*********************************

.. autofunction:: euclidean_distance

.. autofunction:: squared_distance

Abstract Kernels
*********************************

.. autoclass:: AbstractKernel
   :members:

.. autoclass:: CombinationKernel
   :members:

.. autoclass:: AbstractKernelComputation
   :members:

Stationary Kernels
*********************************

.. autoclass:: RBF
   :members:

.. autoclass:: Matern12
   :members:

.. autoclass:: Matern32
   :members:

.. autoclass:: Matern52
   :members:

Nonstationary Kernels
*********************************

.. autoclass:: Polynomial
   :members:


Special Kernels
*********************************

.. autoclass:: GraphKernel
   :members:


Combination Kernels
*********************************

.. autoclass:: SumKernel
   :members:

.. autoclass:: ProductKernel
   :members:

Kernel Computations
*********************************

.. autoclass:: DenseKernelComputation
   :members:

.. autoclass:: DiagonalKernelComputation
   :members:


Likelihoods
#################################

.. automodule:: gpjax.likelihoods
.. currentmodule:: gpjax.likelihoods


Abstract Likelihoods
*********************************

.. autoclass:: AbstractLikelihood
   :members:


Likelihood Functions
*********************************

.. autoclass:: Gaussian
   :members:

.. autoclass:: Bernoulli
   :members:


Mean Functions
#################################

.. automodule:: gpjax.mean_functions
.. currentmodule:: gpjax.mean_functions


Abstract Mean Functions
*********************************

.. autoclass:: AbstractMeanFunction
   :members:


Mean Functions
*********************************

.. autoclass:: Zero
   :members:

.. autoclass:: Constant
   :members:


Sparse Frameworks
#################################

.. automodule:: gpjax.variational_inference
.. currentmodule:: gpjax.variational_inference


Abstract Sparse Objects
*********************************

.. autoclass:: AbstractVariationalInference
   :members:


Sparse Methods
*********************************

.. autoclass:: StochasticVI
   :members:


Variational Families
#################################

.. automodule:: gpjax.variational_families
.. currentmodule:: gpjax.variational_families

Abstract Variational Objects
*********************************

.. autoclass:: AbstractVariationalFamily
   :members:


Gaussian Families
*********************************

.. autoclass:: VariationalGaussian
   :members:

Natural Gradients
*********************************

.. automodule:: gpjax.natural_gradients
.. currentmodule:: gpjax.natural_gradients

.. autofunction:: natural_gradients

.. autofunction:: natural_to_expectation

.. autofunction:: _expectation_elbo

.. autofunction:: _rename_expectation_to_natural

.. autofunction:: _rename_natural_to_expectation

.. autofunction:: _get_moment_trainables

.. autofunction:: _get_hyperparameter_trainables

Datasets
#################################

.. automodule:: gpjax.types
.. currentmodule:: gpjax.types


Dataset
*********************************

.. autoclass:: Dataset
   :members:


Asbtractions
#################################

.. automodule:: gpjax.abstractions
.. currentmodule:: gpjax.abstractions


.. autofunction:: fit

.. autofunction:: fit_batches


Utilities
#################################

.. automodule:: gpjax.utils
.. currentmodule:: gpjax.utils

.. autofunction:: concat_dictionaries

.. autofunction:: sort_dictionary

.. autofunction:: merge_dictionaries

.. autofunction:: dict_array_coercion


Configuration
#################################

.. automodule:: gpjax.config
.. currentmodule:: gpjax.config


.. autofunction:: get_defaults

.. autofunction:: add_parameter


Quadrature
#################################

.. automodule:: gpjax.quadrature
.. currentmodule:: gpjax.quadrature


.. autofunction:: gauss_hermite_quadrature
