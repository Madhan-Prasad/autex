.. autex documentation master file

Welcome to autex's Documentation!
==================================

.. image:: https://img.shields.io/badge/Release-v0.2.0-blue.svg
   :alt: Release

**autex** (*Automated TIFR Extractor*) is a native Python data reduction and extraction
pipeline designed for near-infrared (NIR) imaging and spectroscopy instruments, specifically
the **TIRSPEC** instrument mounted on the 2.0-m Himalayan Chandra Telescope (HCT) at Hanle, Ladakh.

Replacing legacy semi-automated pipelines (e.g., TOIREx), ``autex`` provides fully automated,
deterministic calibration—from raw Non-Destructive Read (NDR) slices up through 2MASS-calibrated
photometry and Horne optimal 1D spectral extraction.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   pipeline_architecture

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/calibration
   api/photometry
   api/astrometry
   api/spectroscopy
   api/io_catalog
   api/pipeline_cli

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
