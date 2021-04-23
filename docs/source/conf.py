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
import re
import os

# using local version of libfmp
import sys
FMP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
assert os.path.exists(os.path.join(FMP_DIR, 'libfmp'))
sys.path.insert(0, FMP_DIR)

import libfmp  # noqa
import libfmp.b  # noqa
import libfmp.c1  # noqa
import libfmp.c2  # noqa
import libfmp.c3  # noqa
import libfmp.c4  # noqa
import libfmp.c5  # noqa
import libfmp.c6  # noqa
import libfmp.c7  # noqa
import libfmp.c8  # noqa

assert libfmp.__path__[0].startswith(FMP_DIR)

# -- Project information -----------------------------------------------------

project = 'libfmp'
copyright = '2021, Meinard Müller and Frank Zalkow'
author = 'Meinard Müller and Frank Zalkow'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.

import pkg_resources  # noqa

libfmp_version = pkg_resources.require('libfmp')[0].version
version = libfmp_version
release = libfmp_version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',   # documentation based on docstrings
    'sphinx.ext.napoleon',  # for having google/numpy style docstrings
    'sphinx.ext.viewcode',  # link source code
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'sphinx.ext.extlinks'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
import sphinx_rtd_theme  # noqa

html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_use_index = True
html_use_modindex = True

html_logo = os.path.join(html_static_path[0], 'Logo_libfmp.png')

html_theme_options = {'logo_only': True}

# do not evaluate keyword default values
# useful, e.g., for libfmp.c6.c6s2_tempo_analysis.compute_plot_tempogram_plp), where np.arange(30, 601) is a default
autodoc_preserve_defaults = True

# Interpret "Returns" section as "Args" section
napoleon_custom_sections = [('Returns', 'params_style'), ('Attributes', 'params_style')]

extlinks = {'fmpbook': ('https://www.audiolabs-erlangen.de/fau/professor/mueller/bookFMP', 'FMP'),
            'fmpnotebook': ('https://www.audiolabs-erlangen.de/resources/MIR/FMP/%s.html', '%s.ipynb')}


# -- Customn pre-processing of docstrings ------------------------------------

def link_notebook(app, what, name, obj, options, lines):
    for i, line in enumerate(lines):
        if 'Notebook:' in line:
            match = re.search('Notebook: (.*?)\.ipynb', line)
            if match:
                link = match.group(1)
                lines[i] = lines[i].replace(f'{link}.ipynb', f':fmpnotebook:`{link}`')


def link_book(app, what, name, obj, options, lines):
    for i, line in enumerate(lines):
        if '[FMP' in line:
            lines[i] = lines[i].replace('[FMP', '[:fmpbook:`\ `')


def remove_module_docstring(app, what, name, obj, options, lines):
    if what == 'module':
        del lines[:]


def setup(app):
    app.connect('autodoc-process-docstring', link_notebook)
    app.connect('autodoc-process-docstring', link_book)
    app.connect('autodoc-process-docstring', remove_module_docstring)
