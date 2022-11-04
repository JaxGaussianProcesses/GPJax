import io
import os
import re

from setuptools import find_packages, setup

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def _parse_requirements(path):

    with open(os.path.join(_CURRENT_DIR, path)) as f:
        return [
            line.rstrip() for line in f if not (line.isspace() or line.startswith("#"))
        ]


# Optional Packages
EXTRAS = {
    "dev": [
        "black",
        "isort",
        "pylint",
        "flake8",
        "pytest",
    ],
    "cuda": ["jax[cuda]"],
}


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
    install_requires=_parse_requirements(
        os.path.join("requirements", "requirements.txt")
    ),
    tests_require=_parse_requirements(
        os.path.join("requirements", "requirements_tests.txt")
    ),
    extras_require=EXTRAS,
    keywords=["gaussian-processes jax machine-learning bayesian"],
)
