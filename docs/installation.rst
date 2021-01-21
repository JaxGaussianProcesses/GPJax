Installation
======================

Stable version
-----------------

To install the latest stable release using :code:`pip` execute the following:

.. code-block:: bash

    pip install gpjax

GPU support
^^^^^^^^^^^^^^^^^^^

GPU support is enabled through proper configuration of the underlying `Jax <https://github.com/google/jax>`_ and `ObJax <https://github.com/google/objax/>`_ installations. CPU enabled forms of both packages are installed as part of the GPJax installation. For GPU Jax support, the following command should be run

.. code-block::bash

    # Specify your installed CUDA version.
    CUDA_VERSION=11.0
    pip install -f https://storage.googleapis.com/jax-releases/jax_releases.html jaxlib==`python3 -c 'import jaxlib; print(jaxlib.__version__)'`+cuda`echo $CUDA_VERSION | sed s:\\\.::g`


Development version
--------------------

To install the latest development version of GPJax, run the following set of commands:

.. code-block:: bash

    git clone https://github.com/thomaspinder/GPJax.git
    cd GPJax
    pip install -r requirements.txt
    python setup.py develop

We recommend that this is done from within a virtual environment.