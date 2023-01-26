import os
import versioneer
from setuptools import find_packages, setup


readme = open("README.md").read()
NAME = "jaxkern"


# Handle builds of nightly release - adapted from BlackJax.
if "BUILD_JAXKERN_NIGHTLY" in os.environ:
    if os.environ["BUILD_JAXKERN_NIGHTLY"] == "nightly":
        NAME += "-nightly"
        from versioneer import get_versions as original_get_versions

        def get_versions():
            from datetime import datetime, timezone

            suffix = datetime.now(timezone.utc).strftime(r".dev%Y%m%d")
            versions = original_get_versions()
            versions["version"] = versions["version"].split("+")[0] + suffix
            return versions

        versioneer.get_versions = get_versions


REQUIRES = [
    "jax>=0.4.1",
    "jaxlib>=0.4.1",
    "jaxutils>=0.0.8",
    "jaxtyping>=0.0.2",
    "jaxlinop>=0.0.3",
    "deprecation",
    "distrax",
]

EXTRAS = {
    "dev": [
        "black",
        "isort",
        "pylint",
        "flake8",
        "pytest",
        "networkx",
        "pytest-cov",
        "pytest-xdist",
    ],
    "cuda": ["jax[cuda]"],
}


setup(
    name=NAME,
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="Daniel Dodd and Thomas Pinder",
    author_email="tompinder@live.co.uk",
    packages=find_packages(".", exclude=["tests"]),
    license="LICENSE",
    description="Kernels in Jax.",
    long_description=readme,
    long_description_content_type="text/markdown",
    project_urls={
        "Documentation": "https://JaxKern.readthedocs.io/en/latest/",
        "Source": "https://github.com/JaxGaussianProcesses/JaxKern",
    },
    install_requires=REQUIRES,
    tests_require=EXTRAS["dev"],
    extras_require=EXTRAS,
    keywords=["gaussian-processes jax machine-learning bayesian"],
)
