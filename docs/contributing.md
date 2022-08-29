---
title: Contributing to GPJax
---

GPJax welcomes contributions from interested individuals or groups.
These guidelines help explain how you can contribute to the library

There are 4 main ways of contributing to the library (in descending
order of difficulty or scope):

- Adding new or improved functionality to the existing codebase
- Fixing outstanding issues (bugs) with the existing codebase. They
  range from low-level software bugs to higher-level design problems.
- Contributing or improving the
  [docs](https://github.com/thomaspinder/GPJax/tree/master/docs) or
  [examples](https://github.com/thomaspinder/GPJax/tree/master/docs/nbs).
- Submitting issues related to bugs or desired enhancements

# Code of conduct

As a contributor to GPJax, you can help us keep the community open and
inclusive. Please read and follow our [Code of
Conduct](https://github.com/thomaspinder/GPJax/blob/master/.github/CODE_OF_CONDUCT.md).

# Opening issues and getting support

Please open issues on [Github Issue
Tracker](https://github.com/thomaspinder/GPJax/issues/new/choose).

You can ask a question or start a discussion in the [Discussion
section](https://github.com/thomaspinder/GPJax/discussions) on Github.

# Contributing code via pull requests

Please submit patches via pull requests.

The preferred workflow for contributing is to fork the [GitHub
repository](https://github.com/thomaspinder/GPJax), clone it to your
local machine, and develop on a feature branch. Once you are ready to
commit your changes, install the pre-commit hooks with
`pre-commit install` and the commit and push your code as usual.

Steps:

1.  Fork the [project repository](https://github.com/thomaspinder/GPJax)
    by clicking on the 'Fork' button near the top right of the main
    repository page. This creates a copy of the code under your GitHub
    user account.

2.  Clone your fork of the GPJax repo from your GitHub account to your
    local disk, and add the base repository as a remote:

    ```bash
    $ git clone git@github.com:<your GitHub handle>/GPJax.git
    $ cd GPJax
    $ git remote add upstream git@github.com:GPJax.git
    ```

3.  Create a `feature` branch to hold your development changes:

    ```bash
    $ git checkout -b my-feature
    ```

    Always use a `feature` branch. It's good practice to never routinely
    work on the `master` branch of any repository.

4.  Project requirements are in `requirements.txt`.

    We suggest using a [virtual
    environment](https://docs.python-guide.org/dev/virtualenvs/) for
    development. Once the virtual environment is activated, run:

    ```bash
    $ pip install -e .
    $ pip install -r requirements-dev.txt
    ```

5.  Install the pre-commit hooks. Please **ensure you do this before
    commiting any files**. This can be done by executing the following:

    ```bash
    $ pre-commit install
    ```

    If successful, this will print the following output
    `pre-commit installed at .git/hooks/pre-commit`.

6.  Develop the feature on your feature branch. Add changed files using
    `git add` and then `git commit` files:

    ```bash
    $ git add modified_files
    $ git commit
    ```

    to record your changes locally. After committing, it is a good idea
    to sync with the base repository in case there have been any
    changes:

    ```bash
    $ git fetch upstream
    $ git rebase upstream/main
    ```

    Then push the changes to your GitHub account with:

    ```bash
    $ git push -u origin my-feature
    ```

7.  Go to the GitHub web page of your fork of the GPJax repo. Click the
    'Pull request' button to send your changes to the project's
    maintainer for review.

# Pull request checklist

We recommended that your contribution complies with the following
guidelines before you submit a pull request:

- If your pull request addresses an issue, please use the pull request
  title to describe the issue and mention the issue number in the pull
  request description. This will make sure a link back to the original
  issue is created.

- All public methods must have informative docstrings

- Please prefix the title of incomplete contributions with `[WIP]` (to
  indicate a work in progress). WIPs may be useful to (1) indicate you
  are working on something to avoid duplicated work, (2) request broad
  review of functionality or API, or (3) seek collaborators.

- All other tests pass when everything is rebuilt from scratch.

- Documentation and high-coverage tests are necessary for enhancements
  to be accepted.

- Code with good test, check with:

  ```bash
  $ pip install -r requirements-dev.txt
  $ pytest tests --cov=./ --cov-report=html
  ```

This guide was derived from [PyMC's guide to
contributing](https://github.com/pymc-devs/pymc/blob/main/CONTRIBUTING.md)
