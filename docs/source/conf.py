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
# import os
# import sys
from datetime import datetime

#
# -- Project information -----------------------------------------------------

project = "Qdrant Client"
copyright = f"{datetime.today().year}, Qdrant Client Authors"
author = "Qdrant Client Authors"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "nbsphinx",
    "IPython.sphinxext.ipython_console_highlighting",
]

# prevents sphinx from adding full path to type hints
autodoc_typehints_format = "short"

# order members by type and not alphabetically, it prevents mixing of class attributes
# and methods
autodoc_member_order = "groupwise"

# moves ``Return type`` to ``Returns``
napoleon_use_rtype = False

# If true, suppress the module name of the python reference if it can be resolved.
# Experimental feature:
# https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-python_use_unqualified_type_names
python_use_unqualified_type_names = True

# prevents sphinx to add full module path in titles
add_module_names = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# prevents unfolding type hints
# autodoc_type_aliases = {
#     "KeyExtractorType": "KeyExtractorType",
#     "TensorInterchange": "TensorInterchange",
#     "CacheCollateReturnType": "CacheCollateReturnType",
#     "CollateFnType": "CollateFnType",
# }

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "*grpc*",
    "*tests*",  # tests are not part of the documentation
]
# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "qdrant_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = []

# Files excluded via exclude_patterns still being generating by sphinx-apidoc
# As they are generated, some documents have links to them. It leads to a warning like:
# `WARNING: toctree contains reference to excluded document '...'`.
# suppress_warnings allows to remove such warnings
suppress_warnings = ["toc.excluded"]

html_theme_options = {
    # google analytics can be added here
    "logo_only": False,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    # Toc options
    "collapse_navigation": True,
    "sticky_navigation": True,
    "titles_only": False,
    "qdrant_project": "qdrant_client",
    "qdrant_logo": "_static/images/logo.svg",
}

# default is false
_FAST_DOCS_DEV = False
