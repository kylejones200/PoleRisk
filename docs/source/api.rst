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

    # Run pole health assessment
    python -m polerisk.main --poles pole_data.csv --soil soil_data.csv
    
    # Launch interactive dashboard
    python dashboard_app.py

For more information, see the main module and dashboard documentation.
