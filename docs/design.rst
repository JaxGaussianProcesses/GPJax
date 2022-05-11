GPJax Principles
======================

GPJax is designed to be a Gaussian process package that provides an accurate representation of the underlying maths. Variable names are designed to closely, if not exactly, match the notation in the :cite:t:`rasmussen2006gaussian`. We here list the notation used in GPJax with its corresponding mathematical quantity.


Gaussian process notation
-----------------

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - On paper
     - GPJax Code
     - Description
   * - :math:`n_\boldsymbol{x}`
     - nx 
     - Number of train inputs
   * - :math:`n_\boldsymbol{t}`
     - nt 
     - Number of test inputs
   * - :math:`\boldsymbol{x} = (x_1,\dotsc,x_{n_\boldsymbol{x}})`
     - x
     - Train inputs
   * - :math:`\boldsymbol{y} = (y_1,\dotsc,y_{n_\boldsymbol{x}})`
     - y
     - Train labels
   * - :math:`\boldsymbol{t} = (t_1,\dotsc,t_{n_\boldsymbol{t}})`
     - t
     - Test inputs
   * - :math:`\boldsymbol{F}`
     - F
     - Latent function modelled as a GP
   * - :math:`\boldsymbol{F}_{\boldsymbol{x}}`
     - Fx 
     - Latent function at inputs :math:`\boldsymbol{x}`
   * - :math:`\boldsymbol{m}_{\boldsymbol{x}}`
     - mx
     - Prior mean at inputs :math:`\boldsymbol{x}`
   * - :math:`\boldsymbol{K}_{\boldsymbol{x}\boldsymbol{x}}`
     - Kxx
     - Kernel Gram matrix at inputs :math:`\boldsymbol{x}`
   * - :math:`\boldsymbol{L}_{\boldsymbol{x}}`
     - Lx 
     - Lower Cholesky decomposition of :math:`\boldsymbol{K}_{\boldsymbol{x}\boldsymbol{x}}`
   * - :math:`\boldsymbol{K}_{\boldsymbol{t}\boldsymbol{x}}`
     - Ktx
     - Cross-covariance between inputs :math:`\boldsymbol{t}` and :math:`\boldsymbol{x}`

Sparse Gaussian process notation
-----------------

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - On paper
     - GPJax Code
     - Description
   * - :math:`n_\boldsymbol{z}`
     - nz
     - Number of inducing inputs
   * - :math:`\boldsymbol{z} = (z_1,\dotsc,z_{n_\boldsymbol{z}})`
     - z
     - Inducing inputs
   * - :math:`\boldsymbol{u} = (u_1,\dotsc,u_{n_\boldsymbol{z}})`
     - u
     - Inducing outputs
