# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import re
import io

from importlib_metadata import version

# sys.path.insert(0, os.path.abspath("../.."))


def read(*names, **kwargs):
    """Function to decode a read files. Credit GPyTorch."""
    with io.open(
        os.path.join(os.path.dirname(__file__), "..", *names),
        encoding=kwargs.get("encoding", "utf8"),
    ) as fp:
        return fp.read()


def find_version(*file_paths):
    """Function to identify the library's current version. Credit GPyTorch."""
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


sys.path.append(
    os.path.abspath(
        os.path.join(
            __file__,
            "..",
            "..",
        )
    )
)


# -- Project information -----------------------------------------------------

project = "GPJax"
copyright = "2021, Thomas Pinder"
author = "Thomas Pinder"

# The full version, including alpha/beta/rc tags
version = find_version("gpjax", "__init__.py")
release = version


# -- General configuration ---------------------------------------------------
# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosectionlabel",
    "sphinx_copybutton",
    "sphinxcontrib.bibtex",
    "sphinxext.opengraph",
]


todo_include_todos = True


bibtex_bibfiles = ["refs.bib"]
bibtex_style = "unsrt"
bibtex_reference_style = "author_year"
nbsphinx_allow_errors = True
nbsphinx_custom_formats = {
    ".pct.py": ["jupytext.reads", {"fmt": "py:percent"}],
}

# Latex commands
# mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
mathjax3_config = {"tex": {"macros": {}}}

with open("latex_symbols.tex", "r") as f:
    for line in f:
        macros = re.findall(r"\\newcommand{\\(.*?)}(\[(\d)\])?{(.+)}", line)
        for macro in macros:
            if len(macro[1]) == 0:
                mathjax3_config["tex"]["macros"][macro[0]] = "{" + macro[3] + "}"
            else:
                mathjax3_config["tex"]["macros"][macro[0]] = ["{" + macro[3] + "}", int(macro[2])]

latex_documents = [
    ("contents", "gpjax.tex", "GPJax", "Thomas Pinder"),
]

latex_engine = "xelatex"  # or 'lualatex'

latex_elements = {}
latex_elements[
    "preamble"
] = r"""
\\usepackage{amsmath,amsfonts,amssymb,amsthm}
"""
latex_appendices = []

# If false, no module index is generated.
latex_use_modindex = True
texinfo_documents = [
    (
        "contents",
        "gpjax",
        "GPJax Documentation",
        "ThomasPinder",
        "GPJax",
        "Jax Gaussian processes",
        "Statistics",
        1,
    ),
]
# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


autosummary_generate = True
napolean_use_rtype = False
autodoc_default_options = {
    "member-order": "bysource",
    "special-members": "__init__, __call__",
    "undoc-members": True,
    "exclude-members": "__weakref__,_abc_impl",
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#B5121B",
        "color-brand-content": "#CC3333",
        "color-admonition-background": "orange",
    },
}


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]
