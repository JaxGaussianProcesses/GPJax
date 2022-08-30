---
title: Installation
---

# Stable version

To install the latest stable release using `pip` execute the following:

```bash
pip install gpjax
```

## GPU support

GPU support is enabled through proper configuration of the underlying
[Jax](https://github.com/google/jax) installation. CPU enabled forms of
both packages are installed as part of the GPJax installation. For GPU
Jax support, the following command should be run

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

# Development version

To install the latest development version of GPJax, run the following
set of commands:

```bash
git clone https://github.com/thomaspinder/GPJax.git
cd GPJax
pip install -r requirements.txt
python setup.py develop
```

We recommend that this is done from within a virtual environment.
