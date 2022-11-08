import nox

# Based on the blog post https://cjolowicz.github.io/posts/hypermodern-python-03-linting/
LOCATIONS = ["gpjax", "tests"]


@nox.session(python=["3.7", "3.8", "3.9", "3.10"])
def lint(session):
    args = session.posargs or LOCATIONS
    session.install("flake8")
    session.run("flake8", *args)
    session.notify("black")


@nox.session(python=["3.7", "3.8", "3.9", "3.10"])
def black(session):
    args = session.posargs or LOCATIONS
    session.install("black")
    session.run("black", *args)
    session.notify("tests")


@nox.session(python=["3.7", "3.8", "3.9", "3.10"])
def tests(session):
    session.install(".")
    session.run("pytest", "-n", "auto")
