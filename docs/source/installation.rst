.. _installation:

Installation
============

This guide will help you install the PoleRisk package and its dependencies.

Prerequisites
------------
- Python 3.12 or higher
- pip (Python package manager)

Installation Methods
-------------------

Using pip (recommended)
~~~~~~~~~~~~~~~~~~~~~~
The easiest way to install the latest stable version is using pip:

.. code-block:: bash

    pip install polerisk

From Source
~~~~~~~~~~~
To install the latest development version directly from the source repository:

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/kylejones200/polerisk.git
    cd polerisk
    
    # Install in development mode
    pip install -e .

Optional Dependencies
---------------------
Install additional capabilities:

.. code-block:: bash

    # For enhanced performance
    pip install polerisk[performance]

    # For machine learning features
    pip install polerisk[ml]

    # For web dashboard
    pip install polerisk[web]

    # For cloud deployment
    pip install polerisk[cloud]

    # Install everything
    pip install polerisk[all]

Dependencies
-----------
The following core dependencies will be installed automatically:

- numpy >= 1.20.0
- pandas >= 1.3.0
- matplotlib >= 3.4.0
- scipy >= 1.7.0
- scikit-learn >= 1.0.0
- signalplot >= 0.1.2
- folium >= 0.12.0

Verification
-----------
To verify the installation, you can run:

.. code-block:: python

    import polerisk
    print(polerisk.__version__)

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

3. If you encounter any other issues, please check the :ref:`troubleshooting` section or open an issue on our `GitHub repository <https://github.com/kylejones200/polerisk/issues>`_.
