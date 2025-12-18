.. _installation:

Installation
============

This guide will help you install the Soil Moisture Analyzer package and its dependencies.

Prerequisites
------------
- Python 3.8 or higher
- pip (Python package manager)

Installation Methods
-------------------

Using pip (recommended)
~~~~~~~~~~~~~~~~~~~~~~
The easiest way to install the latest stable version is using pip:

.. code-block:: bash

    pip install soil-moisture-analyzer

From Source
~~~~~~~~~~~
To install the latest development version directly from the source repository:

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/yourusername/soil-moisture-analyzer.git
    cd soil-moisture-analyzer
    
    # Install in development mode
    pip install -e .

Dependencies
-----------
The following dependencies will be installed automatically:

- numpy >= 1.20.0
- pandas >= 1.3.0
- matplotlib >= 3.4.0
- scipy >= 1.7.0
- netCDF4 >= 1.5.0
- cartopy >= 0.19.0
- seaborn >= 0.11.0

Verification
-----------
To verify the installation, you can run:

.. code-block:: python

    import soilmoisture
    print(soilmoisture.__version__)

This should print the installed version of the package without any errors.

Troubleshooting
--------------
If you encounter any issues during installation, please check the following:

1. Ensure you have the latest version of pip:
   .. code-block:: bash

       pip install --upgrade pip

2. On some systems, you might need to install system-level dependencies for Cartopy:
   
   - For Ubuntu/Debian:
     .. code-block:: bash

         sudo apt-get install libgeos-dev libproj-dev proj-data proj-bin

   - For macOS (using Homebrew):
     .. code-block:: bash

         brew install geos proj

3. If you encounter any other issues, please check the :ref:`troubleshooting` section or open an issue on our `GitHub repository <https://github.com/yourusername/soil-moisture-analyzer/issues>`_.
