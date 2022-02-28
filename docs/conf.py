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

# sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------

project = "GPJax"
copyright = "2021, Thomas Pinder"
author = "Thomas Pinder"

# The full version, including alpha/beta/rc tags
release = "0.4"


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
]

### Automatic API doc generation
extensions.append("autoapi.extension")
autoapi_dirs = ["../gpjax"]
autoapi_add_toctree_entry = False
autoapi_python_class_content = "both"
autoapi_options = [
    "members",
    "private-members",
    "special-members",
    "imported-members",
    "show-inheritance",
]

bibtex_bibfiles = ["refs.bib"]
bibtex_style = "unsrt"
bibtex_reference_style = "author_year"
nbsphinx_allow_errors = True
nbsphinx_custom_formats = {
    ".pct.py": ["jupytext.reads", {"fmt": "py:percent"}],
}

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
html_static_path = ["_static"]
