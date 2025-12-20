.. _api:

API Reference
============

This page provides detailed documentation for the PoleRisk API.

Core Modules
-----------

.. toctree::
   :maxdepth: 2

   modules/polerisk

.. automodule:: polerisk
   :members:
   :undoc-members:
   :show-inheritance:

Pole Health Assessment
---------------------

.. automodule:: polerisk.pole_health.assessment
   :members:
   :undoc-members:
   :show-inheritance:

Risk Scoring
------------

.. automodule:: polerisk.pole_health.risk_scoring
   :members:
   :undoc-members:
   :show-inheritance:

Visualization
-------------

.. automodule:: polerisk.visualization
   :members:
   :undoc-members:
   :show-inheritance:

Database
--------

.. automodule:: polerisk.database
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
