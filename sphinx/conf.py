#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# komm documentation build configuration file, created by
# sphinx-quickstart on Fri Nov 10 13:44:30 2017.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import datetime
import os
import sys

sys.path.insert(0, os.path.abspath("../"))

from komm import __version__

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
needs_sphinx = "1.7"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinxcontrib.bibtex",
]

add_module_names = False

# Autosummary settings
autosummary_generate = True

# Napoleon settings
napoleon_use_admonition_for_examples = True
napoleon_use_rtype = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ["templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "Komm"
author = "Roberto W. Nobrega"
copyright = f"2018–{datetime.date.today().year}, Roberto W. Nobrega"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = __version__
# The full version, including alpha/beta/rc tags.
release = version

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False


# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "nature"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {
#    'navigation_depth': 2,
# }

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["static"]

html_css_files = ["custom.css"]

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# This is required for the alabaster theme
# refs: http://alabaster.readthedocs.io/en/latest/installation.html#sidebars
html_sidebars = {
    "**": [
        "libref.html",
        "indices.html",
        "relations.html",  # needs 'show_related': True theme option to display
        "sourcelink.html",
        "searchbox.html",
    ]
}

# Bibtex
bibtex_bibfiles = ["refs.bib"]

# -------------------------------------------------------------------------


def builder_inited_handler(app):
    print("Removing old _build...")
    os.system("rm -rf _build/")
    print("Converting PDF to PNG...")
    for filename in os.listdir("figures"):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join("figures", filename)
            noext_path = pdf_path[:-4]
            png_path = noext_path + ".png"
            if not os.path.isfile(png_path):
                os.system("pdftoppm -cropbox -singlefile -png -r 75 {} {}".format(pdf_path, noext_path))


def setup(app):
    app.connect("builder-inited", builder_inited_handler)
