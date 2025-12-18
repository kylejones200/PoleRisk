.. _usage:

Usage Guide
==========

This guide provides an overview of how to use the Soil Moisture Analyzer package.

Basic Usage
-----------

1. **Import the package**:

   .. code-block:: python

      import soilmoisture
      from soilmoisture.data import load_data
      from soilmoisture.visualization import plot_time_series

2. **Load your data**:

   .. code-block:: python

      # Load in-situ and satellite data
      data = load_data(
          in_situ_path='path/to/insitu_data.csv',
          satellite_path='path/to/satellite_data.nc'
      )

3. **Process the data**:

   .. code-block:: python

      # Process and match the data
      matched_data = soilmoisture.match_data(data)

4. **Visualize the results**:

   .. code-block:: python

      # Create a time series plot
      fig = plot_time_series(
          matched_data,
          x='date',
          y=['in_situ', 'satellite'],
          title='Soil Moisture Time Series'
      )
      fig.savefig('soil_moisture_plot.png')

Command Line Interface
---------------------

The package provides command-line tools for common tasks:

1. **Process data**:

   .. code-block:: bash

      soilmoisture-analyze --input input_data.nc --output results.csv

2. **Generate visualizations**:

   .. code-block:: bash

      soilmoisture-visualize --input results.csv --output-dir plots/

Example Workflow
----------------

Here's a complete example workflow:

.. code-block:: python

   import os
   import pandas as pd
   import soilmoisture
   from soilmoisture.data import load_data, process_data
   from soilmoisture.visualization import (
       plot_time_series,
       plot_scatter,
       plot_distributions,
       create_dashboard
   )

   # Set up paths
   input_dir = 'Input'
   output_dir = 'Output'
   os.makedirs(output_dir, exist_ok=True)

   # Load and process data
   data = load_data(
       in_situ_path=os.path.join(input_dir, 'In-situ data/insitu_measurements.csv'),
       satellite_path=os.path.join(input_dir, 'LPRM_NetCDF/satellite_data.nc')
   )

   # Process and match the data
   matched_data = soilmoisture.match_data(data)

   # Save results
   output_file = os.path.join(output_dir, 'matched_results.csv')
   matched_data.to_csv(output_file, index=False)

   # Generate visualizations
   plot_time_series(
       matched_data,
       output_dir=output_dir,
       title='Soil Moisture Time Series'
   )

   plot_scatter(
       matched_data,
       output_dir=output_dir,
       x='in_situ',
       y='satellite',
       title='In-situ vs Satellite Soil Moisture'
   )

   # Create an interactive dashboard
   dashboard_path = create_dashboard(matched_data, output_dir=output_dir)
   print(f"Dashboard created at: {dashboard_path}")

Configuration
-------------

You can configure the package using a configuration file (``config.ini``):

.. code-block:: ini

   [paths]
   input_dir = Input
   output_dir = Output
   
   [processing]
   time_zone = America/New_York
   resample_freq = 1D
   
   [visualization]
   style = seaborn
   figsize = 12, 8
   dpi = 300

Then load the configuration in your code:

.. code-block:: python

   from soilmoisture.config import load_config
   
   config = load_config('config.ini')
   print(f"Input directory: {config['paths']['input_dir']}")

Troubleshooting
---------------

Common issues and solutions:

1. **Missing data files**:
   - Ensure all required input files are in the correct directories
   - Check file permissions and paths in your configuration

2. **Visualization issues**:
   - Make sure all required dependencies are installed
   - Check that your data contains the expected columns

3. **Performance problems**:
   - For large datasets, consider downsampling or using a subset of the data
   - Close other memory-intensive applications

For additional help, please refer to the :ref:`api` documentation or open an issue on our `GitHub repository <https://github.com/yourusername/soil-moisture-analyzer/issues>`_.
