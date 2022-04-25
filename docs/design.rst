GPJax Principles
======================

GPJax is designed to be a Gaussian process package that provides an accurate representation of the underlying maths.

Notation
-----------------

Variable names are designed to closely, if not exactly, match the notation in the :cite:t:`rasmussen2006gaussian`. We here list the notation used in GPJax with its corresponding mathematical quantity.

:math:`x` : Training inputs
:math:`y` : Training labels
:math:`t` : Testing inputs

:math:`F` : Latent function being modelled as a GP
:math:`Fx`: latent function, :math:`F` , at train inputs, :math:`x`
:math:`Fmu` :  Predictive mean of the latent function, :math:`F`
:math:`Fcov` : Predictive covariance of the latent function, :math:`F`
:math:`Fvar` :  predictive (diagonal) variance of the latent function, :math:`F`

:math:`nx` :  Number of train inputs, :math:`x`
:math:`Kxx` : Kernel's Gram matrix at train inputs, :math:`x`
:math:`Lx` : Lower cholesky decomposition at train inputs, :math:`x`
:math:`mx` : Prior mean at train inputs, :math:`x`

:math:`nt` : Number of test inputs, :math:`t`
:math:`Ktt` : Kernel's Gram matrix at test inputs, :math:`t`
:math:`Lt` : Lower cholesky decomposition at test inputs, :math:`t`
:math:`mt` : Prior mean at test inputs, :math:`t`

:math:`Ktx` : Cross covariance between test inputs, :math:`t`, and train inputs, :math:`x`

:math:`z` : Inducing inputs
:math:`u` : Inducing outputs
:math:`nz` : Number of inducing inputs, :math:`z`
:math:`Kzz` : Kernel's Gram matrix at inducing inputs, :math:`z`
:math:`Lz` : Lower cholesky decomposition at inducing inputs, :math:`z`
:math:`Kzx` : Cross covariance between test inputs, :math:`z`, and train inputs, :math:`x`

:math:`q(u) = \mathcal{N}[u; \mu,  sqrt.sqrt^T]`, is the standard (whiten is false).
:math:`q(u) = N[u; Lz.mu + mz`,  :math:`(Lz.sqrt).(Lz.sqrt)^T]`, is the whitened parameterisation.
