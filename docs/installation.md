# Installation

## Stable version

The latest stable release of `GPJax` can be installed via `pip`:

```bash
pip install gpjax
```

!!! note "Check your installation"
    We recommend you check your installation version:
    ```
    python -c 'import gpjax; print(gpjax.__version__)'
    ```

## GPU/TPU support

Fancy using GPJax on GPU/TPU? Then you'll need to install JAX with the relevant
hardware acceleration support as detailed in the
[JAX installation guide](https://github.com/google/jax#installation).


## Development version

!!! warning
    This version is possibly unstable and may contain bugs.

    The latest development version of `GPJax` can be installed via running following:

    ```bash
    git clone https://github.com/thomaspinder/GPJax.git
    cd GPJax
    hatch shell create
    ```

!!! tip
    We advise you create virtual environment before installing:

    ```bash
    conda create -n gpjax_experimental python=3.10.0
    conda activate gpjax_experimental
    ```

    and recommend you check your installation passes the supplied unit tests:

    ```bash
    hatch run dev:all-tests
    ```
