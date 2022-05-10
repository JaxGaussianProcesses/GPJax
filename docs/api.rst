.. role:: hidden
    :class: hidden-section

**********************
Package Reference
**********************

Gaussian Processes
#################################

.. automodule:: gpjax.gps
.. currentmodule:: gpjax.gps


Abstract GPs
*********************************

.. autoclass:: AbstractGP
   :members:

.. autoclass:: AbstractPosterior
   :members:


Gaussian Process Priors
*********************************

.. autoclass:: Prior
   :members:

Gaussian Process Posteriors
*********************************

.. autoclass:: ConjugatePosterior
   :members:

.. autoclass:: NonConjugatePosterior
   :members:


Kernels
########################

.. automodule:: gpjax.kernels
.. currentmodule:: gpjax.kernels

Kernel Functions
*********************************

.. autofunction:: gram

.. autofunction:: cross_covariance

.. autofunction:: diagonal

.. autofunction:: euclidean_distance

.. autofunction:: squared_distance

Abstract Kernels
*********************************

.. autoclass:: Kernel
   :members:

.. autoclass:: CombinationKernel
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

.. autoclass:: MeanFunction
   :members:


Mean Functions
*********************************

.. autoclass:: Zero
   :members:

.. autoclass:: Constant
   :members:


Sparse Frameworks
#################################

.. automodule:: gpjax.sparse_gps
.. currentmodule:: gpjax.sparse_gps


Abstract Sparse Objects
*********************************

.. autoclass:: VariationalPosterior
   :members:


Sparse Methods
*********************************

.. autoclass:: SVGP
   :members:


Variational Families
#################################

.. automodule:: gpjax.variational
.. currentmodule:: gpjax.variational

Abstract Variational Objects
*********************************

.. autoclass:: VariationalFamily
   :members:


Gaussian Families
*********************************

.. autoclass:: VariationalGaussian
   :members:


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

.. autofunction:: optax_fit

.. autofunction:: fit_batches


Utilities
#################################

.. automodule:: gpjax.utils
.. currentmodule:: gpjax.utils

.. autofunction:: I

.. autofunction:: concat_dictionaries

.. autofunction:: sort_dictionary

.. autofunction:: as_constant

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
