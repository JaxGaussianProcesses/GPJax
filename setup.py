import io
import os
import re

from setuptools import find_packages, setup


# Get version number (function from GPyTorch)
def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8"),
    ) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


version = find_version("gpjax", "__init__.py")
readme = open("README.md").read()


REQUIRES = [
    "jax>=0.1.67",
    "jaxlib>=0.1.47",
    "optax",
    "chex",
    "distrax>=0.1.2",
    "tqdm>=4.0.0",
    "ml-collections==0.1.0",
    "jaxtyping>=0.0.2",
    "jaxlinop>=0.0.3",
]

EXTRAS = {
    "dev": ["black", "isort", "pylint", "flake8", "pytest", "networkx", "pytest-cov"],
    "cuda": ["jax[cuda]"],
}


setup(
    name="GPJax",
    version=version,
    author="Thomas Pinder",
    author_email="t.pinder2@lancaster.ac.uk",
    packages=find_packages(".", exclude=["tests"]),
    license="LICENSE",
    description="Didactic Gaussian processes in Jax.",
    long_description=readme,
    long_description_content_type="text/markdown",
    project_urls={
        "Documentation": "https://gpjax.readthedocs.io/en/latest/",
        "Source": "https://github.com/thomaspinder/GPJax",
    },
    install_requires=REQUIRES,
    tests_require=EXTRAS["dev"],
    extras_require=EXTRAS,
    keywords=["gaussian-processes jax machine-learning bayesian"],
)
