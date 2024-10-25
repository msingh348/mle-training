# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "mle-training"
copyright = "2024, Mohan Singh"
author = "Mohan Singh"
release = "0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Add extensions
extensions = [
    "sphinx.ext.autodoc",  # Automatically document from docstrings
    "sphinx.ext.napoleon",  # For NumPy and Google style docstrings
]

# Use the Napoleon extension to parse NumPy-style docstrings
napoleon_numpy_docstring = True

# Paths
templates_path = ["_templates"]
exclude_patterns = []

# Language setting ('en' for English)
language = "en"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]
