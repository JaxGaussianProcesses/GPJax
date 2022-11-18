"""Sphinx configuration."""
project = "JaxKern"
author = "Thomas Pinder and Daniel Dodd"
copyright = "2022, Thomas Pinder and Daniel Dodd"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_click",
    "myst_parser",
]
autodoc_typehints = "description"
html_theme = "furo"
