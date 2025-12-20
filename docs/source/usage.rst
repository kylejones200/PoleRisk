.. _usage:

Usage Guide
==========

This guide provides an overview of how to use the PoleRisk package for utility pole health assessment and maintenance optimization.

Basic Usage
-----------

1. **Import the package**:

   .. code-block:: python

      import polerisk
      from polerisk.pole_health import PoleHealthAssessment
      from polerisk.pole_health.risk_scoring import PoleRiskScorer

2. **Load your pole data**:

   .. code-block:: python

      # Load pole inventory data
      import pandas as pd
      poles_df = pd.read_csv('pole_inventory.csv')

3. **Assess pole health**:

   .. code-block:: python

      # Create assessment instance
      assessor = PoleHealthAssessment()
      
      # Assess all poles
      assessment_results = assessor.assess_poles(poles_df)

4. **Calculate risk scores**:

   .. code-block:: python

      # Calculate risk scores
      risk_scorer = PoleRiskScorer()
      risk_scores = risk_scorer.calculate_risk(assessment_results)
      
      # Identify high-risk poles
      high_risk_poles = risk_scores[risk_scores['risk_score'] > 0.7]

Example Workflow
----------------

Here's a complete example workflow:

.. code-block:: python

   import pandas as pd
   from polerisk.pole_health import PoleHealthAssessment
   from polerisk.pole_health.risk_scoring import PoleRiskScorer, MaintenanceScheduler

   # Load pole data
   poles_df = pd.read_csv('Input/sample_poles.csv')
   soil_df = pd.read_csv('Input/sample_soil_data.csv')

   # Assess pole health
   assessor = PoleHealthAssessment()
   assessment = assessor.assess_poles(poles_df, soil_data=soil_df)

   # Calculate risk scores
   risk_scorer = PoleRiskScorer()
   risk_assessment = risk_scorer.calculate_risk(assessment)

   # Generate maintenance schedule
   scheduler = MaintenanceScheduler()
   maintenance_schedule = scheduler.optimize_schedule(
       risk_assessment,
       budget=100000,
       time_horizon_days=365
   )

   # Save results
   assessment.to_csv('Output/pole_health_assessment.csv', index=False)
   maintenance_schedule.to_csv('Output/maintenance_schedule.csv', index=False)

   print(f"Assessed {len(assessment)} poles")
   print(f"High-risk poles: {len(risk_assessment[risk_assessment['risk_score'] > 0.7])}")

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

For additional help, please refer to the :ref:`api` documentation or open an issue on our `GitHub repository <https://github.com/kylejones200/polerisk/issues>`_.
