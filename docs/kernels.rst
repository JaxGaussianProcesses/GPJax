.. role:: hidden
    :class: hidden-section

Kernels
===================================

.. automodule:: gpjax.kernels
.. currentmodule:: gpjax.kernels

Kernel Functions
-----------------------------

.. autofunction:: gram

.. autofunction:: cross_covariance

.. autofunction:: diagonal

.. autofunction:: euclidean_distance

.. autofunction:: squared_distance

Abstract Kernels
-----------------------------

.. autoclass:: Kernel
   :members:

.. autoclass:: CombinationKernel
   :members:


Stationary Kernels
-----------------------------

.. autoclass:: RBF
   :members:

.. autoclass:: Matern12
   :members:

.. autoclass:: Matern32
   :members:

.. autoclass:: Matern52
   :members:

Nonstationary Kernels
-----------------------------

.. autoclass:: Polynomial
   :members:


Special Kernels
-----------------------------

.. autoclass:: GraphKernel
   :members:


Combination Kernels
-----------------------------

.. autoclass:: SumKernel
   :members:

.. autoclass:: ProductKernel
   :members:

