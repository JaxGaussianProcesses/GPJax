import nox
import pathlib
import os

# Based on the blog post https://cjolowicz.github.io/posts/hypermodern-python-03-linting/
LOCATIONS = ["gpjax", "tests"]
VENV_DIR = pathlib.Path("./.venv").resolve()


@nox.session(python="3.9")
def lint(session):
    args = session.posargs or LOCATIONS
    session.install("flake8")
    session.run("flake8", *args)


@nox.session(python="3.9")
def black(session):
    args = session.posargs or LOCATIONS
    session.install("black")
    session.run("black", *args)


@nox.session(python="3.9")
def pydocstyle(session):
    """
    Nox run pydocstyle
    Args:
        session: nox session
    Returns:
        None
    Raises:
        N/A
    """
    session.install(f"pydocstyle")
    session.run("python", "-m", "pydocstyle", ".")


@nox.session(python="3.9")
def tests(session: nox.session):
    args = session.posargs or LOCATIONS
    session.run("python", "-m", "pip", "install", "-e", ".[dev]")
    session.install(".")
    session.run("pytest", "-n", "auto", "--cov", "./", "--cov-report", "xml", *args)
