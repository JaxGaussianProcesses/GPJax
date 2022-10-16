# Installation

## Stable version

The latest stable release of `GPJax` can be installed via ``pip``:

```bash
pip install gpjax
```

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

The latest development version of `GPJax` can be installed via running following:

```bash
git clone https://github.com/thomaspinder/GPJax.git
cd GPJax
pip install -r requirements.txt
python setup.py develop
```

We recommend that this is done from within a virtual environment.
