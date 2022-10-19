# Installation

## Stable version

The latest stable release of `GPJax` can be installed via `pip`:

```bash
pip install gpjax
```

:::{note}
We recommend you check your installation version:
```
python -c 'import gpjax; print(gpjax.__version__)'
```
:::

## GPU support

GPU support is enabled through proper configuration of the underlying
[Jax](https://github.com/google/jax) installation. CPU enabled forms of
both packages are installed as part of the `GPJax` installation. For GPU
Jax support, the following commands should be run:

```bash
# Specify your installed CUDA version.
CUDA_VERSION=11.0
pip install jaxlib
```

Then, within a Python shell run

```python
import jaxlib
print(jaxlib.__version__)
```

## Development version

:::{warning}
This version is possibly unstable and may contain bugs.
:::

The latest development version of `GPJax` can be installed via running following:

```bash
git clone https://github.com/thomaspinder/GPJax.git
cd GPJax
python setup.py develop
```
:::{tip}
We advise you create virtual environment before installing:

```bash
conda create -n gpjax_experimental python=3.10.0
conda activate gpjax_experimental
```

and recommend you check your installation passes the supplied unit tests:

```bash
python -m pytest tests/
```
:::