import os
import versioneer
from setuptools import find_packages, setup


NAME = "gpjax"
README = open("README.md").read()

# Handle builds of nightly release
if "BUILD_GPJAX_NIGHTLY" in os.environ:
    if os.environ["BUILD_GPJAX_NIGHTLY"] == "nightly":
        NAME += "-nightly"

        from versioneer import get_versions as original_get_versions

        def get_versions():
            from datetime import datetime, timezone

            suffix = datetime.now(timezone.utc).strftime(r".dev%Y%m%d")
            versions = original_get_versions()
            versions["version"] = versions["version"].split("+")[0] + suffix
            return versions

        versioneer.get_versions = get_versions


EXTRAS = {
    "dev": [
        "black",
        "isort",
        "pylint",
        "flake8",
        "pytest",
        "networkx",
        "pytest-cov",
    ],
    "cuda": ["jax[cuda]"],
}


setup(
    name=NAME,
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="Thomas Pinder",
    author_email="t.pinder2@lancaster.ac.uk",
    packages=find_packages(".", exclude=["tests"]),
    license="LICENSE",
    description="Didactic Gaussian processes in Jax.",
    long_description=README,
    long_description_content_type="text/markdown",
    project_urls={
        "Documentation": "https://gpjax.readthedocs.io/en/latest/",
        "Source": "https://github.com/thomaspinder/GPJax",
    },
    python_requires=">=3.7",
    install_requires=[
        "jax>=0.4.1",
        "jaxlib>=0.4.1",
        "optax",
        "jaxutils>=0.0.6",
        "jaxkern>=0.0.4",
        "distrax>=0.1.2",
        "tqdm>=4.0.0",
        "ml-collections==0.1.0",
        "jaxtyping>=0.0.2",
        "jaxlinop>=0.0.3",
        "deprecation",
    ],
    tests_require=EXTRAS["dev"],
    extras_require=EXTRAS,
    keywords=["gaussian-processes jax machine-learning bayesian"],
)
