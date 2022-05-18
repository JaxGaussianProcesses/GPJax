GPJax Principles
======================

GPJax is designed to be a Gaussian process package that provides an accurate representation of the underlying maths. Variable names are designed to closely, if not exactly, match the notation in the :cite:t:`rasmussen2006gaussian`. We here list the notation used in GPJax with its corresponding mathematical quantity.


Gaussian process notation
-----------------

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - On paper
     - GPJax code
     - Description
   * - :math:`n`
     - n 
     - Number of train inputs
   * - :math:`\boldsymbol{x} = (x_1,\dotsc,x_{n})`
     - x
     - Train inputs
   * - :math:`\boldsymbol{y} = (y_1,\dotsc,y_{n})`
     - y
     - Train labels
   * - :math:`\boldsymbol{t}`
     - t
     - Test inputs
   * - :math:`f(\cdot)`
     - f
     - Latent function modelled as a GP
   * - :math:`f({\boldsymbol{x}})`
     - fx
     - Latent function at inputs :math:`\boldsymbol{x}`
   * - :math:`\boldsymbol{\mu}_{\boldsymbol{x}}`
     - μx
     - Prior mean at inputs :math:`\boldsymbol{x}`
   * - :math:`\mathbf{K}_{\boldsymbol{x}\boldsymbol{x}}`
     - Kxx
     - Kernel Gram matrix at inputs :math:`\boldsymbol{x}`
   * - :math:`\mathbf{L}_{\boldsymbol{x}}`
     - Lx 
     - Lower Cholesky decomposition of :math:`\boldsymbol{K}_{\boldsymbol{x}\boldsymbol{x}}`
   * - :math:`\mathbf{K}_{\boldsymbol{t}\boldsymbol{x}}`
     - Ktx
     - Cross-covariance between inputs :math:`\boldsymbol{t}` and :math:`\boldsymbol{x}`

Sparse Gaussian process notation
-----------------

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - On paper
     - GPJax code
     - Description
   * - :math:`m`
     - m
     - Number of inducing inputs
   * - :math:`\boldsymbol{z} = (z_1,\dotsc,z_{m})`
     - z
     - Inducing inputs
   * - :math:`\boldsymbol{u} = (u_1,\dotsc,u_{m})`
     - u
     - Inducing outputs
