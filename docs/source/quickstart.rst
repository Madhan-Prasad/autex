Quickstart Guide
================

Command-Line Usage
------------------

Run Slope Fitting on Raw SUTR Ramps:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: bash

   autex /path/to/raw/data --mode slope --output data/reduced

Run 2MASS-Calibrated Photometry:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: bash

   autex /path/to/calibrated_slopes --mode photometry --output data/reduced

Run ABBA Nod-Pair Spectroscopy:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: bash

   autex /path/to/nod_frames --mode spectroscopy --output data/reduced

Python API Quickstart
---------------------

.. code-block:: python

   from autex.config import Config
   from autex.pipeline import ReductionPipeline

   config = Config.from_yaml("configs/default_config.yaml")
   pipeline = ReductionPipeline(config)

   # Execute full reduction workflow
   pipeline.run("data/raw", mode="all")
