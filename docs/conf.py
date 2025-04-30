# -*- coding: utf-8 -*-

import os
import thecannon as tc

language = "en"

extensions = [
    "numpydoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
    # "breathe",
]
intersphinx_mapping = {
   'python': ('https://docs.python.org/3/', None),
   'numpy': ('https://docs.scipy.org/doc/numpy/', None),
#    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
#    'astropy': ('http://docs.astropy.org/en/stable/', None),
#    'emcee': ('http://dan.iel.fm/emcee/current/', None)
}

breathe_projects = {"celerite": "xml"}
breathe_default_project = "celerite"

templates_path = ["_templates"]
source_suffix = ".rst"
master_doc = "index"

napoleon_numpy_docstring = True

autosummary_generate = False

# General information about the project.
project = "The Cannon"
author = " & contributors"
copyright = "2016, 2017, 2025" + author

version = "1.0"
release = "1.0.0a"

exclude_patterns = ["_build"]
pygments_style = "sphinx"

# Readthedocs.
# on_rtd = os.environ.get("READTHEDOCS", None) == "True"
# if on_rtd:
#     import subprocess
#     subprocess.call("doxygen Doxyfile", shell=True)
# else:
import sphinx_rtd_theme
html_theme = "sphinx_rtd_theme"
    # html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

html_context = dict(
    display_github=True,
    github_user="andycasey",
    github_repo="AnniesLasso",
    github_version="refactor",
    conf_py_path="/docs/",
)
html_static_path = ["_static"]
html_show_sourcelink = False
