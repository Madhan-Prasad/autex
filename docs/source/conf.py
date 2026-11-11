"""Sphinx configuration for autex (Automated TIFR Extractor)."""

import os
import sys
from pathlib import Path

# Add src to sys.path so autodoc can find autex
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

project = "autex"
copyright = "2026, NIUS Research Group (TIFR / HCT)"
author = "Madhan & NIUS Collaboration"
version = "0.2.0"
release = "0.2.0"

# Core Extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",       # Support Google and NumPy style docstrings
    "sphinx.ext.viewcode",       # Add links to highlighted source code
    "sphinx.ext.mathjax",        # Render LaTeX equations
    "sphinx.ext.intersphinx",    # Link to external docs (numpy, astropy)
    "sphinx_autodoc_typehints",  # Automatically document type hints
    "myst_parser",               # Support Markdown files (.md)
]

# Source file suffixes
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Napoleon configuration for scientific docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

# HTML Theme
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 3,
    "includehidden": True,
    "titles_only": False,
}

# InterSphinx mappings to link numpy and astropy classes
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
}
