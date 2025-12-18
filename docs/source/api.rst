.. _api:

API Reference
============

This page provides detailed documentation for the Soil Moisture Analyzer API.

Core Modules
-----------

.. toctree::
   :maxdepth: 2

   modules/soilmoisture

.. automodule:: soilmoisture
   :members:
   :undoc-members:
   :show-inheritance:

Data Processing
---------------

.. automodule:: soilmoisture.data
   :members:
   :undoc-members:
   :show-inheritance:

Visualization
-------------

.. automodule:: soilmoisture.visualization
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: soilmoisture.visualization.plots
   :members:
   :undoc-members:
   :show-inheritance:

Utilities
---------

.. automodule:: soilmoisture.utils
   :members:
   :undoc-members:
   :show-inheritance:

Command Line Interface
----------------------

The package provides command-line tools for common tasks:

.. code-block:: bash

    # Process soil moisture data
    soilmoisture-analyze --input input.nc --output results.csv
    
    # Generate visualizations
    soilmoisture-visualize --input results.csv --output-dir plots/

For more information on the available options, run:

.. code-block:: bash

    soilmoisture-analyze --help
    soilmoisture-visualize --help
