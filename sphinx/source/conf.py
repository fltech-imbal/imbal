import os
import sys

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


project = 'imbal'
copyright = '2025, Dr. Philip Chan, Thomas Galletta'
author = 'Dr. Philip Chan, Thomas Galletta'
release = ''

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'myst_parser'
]

templates_path = ['_templates']
exclude_patterns = []

sys.path.insert(0, os.path.abspath('../../'))  # Adjust path as needed


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

autodoc_member_order = 'groupwise'

autodoc_default_options  = {
    'members' : True,
    'inherited-members' : True,
    'show-inheritance' : True,
    'undoc-members' : True
}

