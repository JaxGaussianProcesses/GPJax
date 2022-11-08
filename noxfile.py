import nox

# Based on the blog post https://cjolowicz.github.io/posts/hypermodern-python-03-linting/
LOCATIONS = ["gpjax", "tests"]


@nox.session
def tests(session):
    session.install(".")
    session.run("pytest", "-n", "auto")


@nox.session(python="3.8")
def lint(session):
    args = session.posargs or LOCATIONS
    session.install(
        "flake8",
        "flake8-bandit",
        "flake8-black",
        "flake8-bugbear",
        "flake8-import-order",
    )
    session.run("flake8", *args)


@nox.session(python="3.8")
def black(session):
    args = session.posargs or LOCATIONS
    session.install("black")
    session.run("black", *args)
