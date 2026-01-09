#!/usr/bin/env python3
"""
Example: Failure Mode Analysis

Demonstrates how to use the failure mode model to assess specific failure modes
for utility poles.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from polerisk.pole_health import (
    PoleInfo,
    FailureModeModel,
    FailureModeAnalysis,
)
from polerisk.pole_health.pole_data import PoleHealthMetrics
from datetime import datetime


def create_sample_pole() -> PoleInfo:
    """Create a sample pole for demonstration."""
    return PoleInfo(
        pole_id="POL-001",
        latitude=35.0,
        longitude=-97.0,
        install_date=datetime(1995, 1, 1),
        pole_type="wood",
        material="southern_pine",
        height_ft=45.0,
        voltage_class="distribution",
        structure_type="tangent",
    )


def create_sample_health_metrics() -> PoleHealthMetrics:
    """Create sample health metrics."""
    return PoleHealthMetrics(
        pole_id="POL-001",
        assessment_date=datetime.now(),
        overall_health_score=65.0,
        soil_stability_score=55.0,
        structural_risk_score=70.0,
        moisture_risk=0.45,
        erosion_risk=0.30,
        chemical_corrosion_risk=0.25,
        freeze_thaw_risk=0.20,
        bearing_capacity_risk=0.35,
        predicted_failure_probability=0.15,
    )


def main():
    """Run failure mode analysis example."""
    print("=" * 60)
    print("Failure Mode Analysis Example")
    print("=" * 60)
    print()
    
    # Create sample data
    pole = create_sample_pole()
    health_metrics = create_sample_health_metrics()
    
    # Environmental data (would come from real sources)
    environmental_data = {
        'wind_exposure': 0.7,  # High wind exposure
        'road_proximity': 0.3,  # Moderate road proximity
        'wildfire_risk': 0.2,  # Low wildfire risk
        'ice_zone': 0.1,  # Low ice risk
        'drought_index': 0.6,  # Moderate drought conditions
    }
    
    # Initialize failure mode model
    model = FailureModeModel()
    
    # Perform failure mode analysis
    print(f"Analyzing failure modes for pole: {pole.pole_id}")
    print(f"Age: {pole.age_years:.1f} years" if pole.age_years else "Age: Unknown")
    print(f"Material: {pole.material}")
    print(f"Health Score: {health_metrics.overall_health_score:.2f}")
    print()
    
    analysis = model.assess_failure_modes(pole, health_metrics, environmental_data)
    
    # Display results
    print("=" * 60)
    print("FAILURE MODE ANALYSIS RESULTS")
    print("=" * 60)
    print()
    print(f"Primary Failure Mode: {analysis.primary_mode.value.upper().replace('_', ' ')}")
    print(f"Overall Risk: {analysis.overall_risk:.2%}")
    print()
    
    print("Failure Mode Risks (sorted by probability):")
    print("-" * 60)
    for i, mode_risk in enumerate(analysis.mode_risks[:5], 1):  # Top 5
        print(f"\n{i}. {mode_risk.mode.value.upper().replace('_', ' ')}")
        print(f"   Probability: {mode_risk.probability:.2%}")
        print(f"   Confidence: {mode_risk.confidence:.2%}")
        
        if mode_risk.time_to_failure_estimate:
            lower, upper = mode_risk.time_to_failure_confidence
            print(f"   Estimated Time to Failure: {mode_risk.time_to_failure_estimate:.1f} years")
            print(f"   Confidence Range: {lower:.1f} - {upper:.1f} years")
        
        print(f"   Top Contributing Factors:")
        sorted_factors = sorted(
            mode_risk.contributing_factors.items(),
            key=lambda x: x[1],
            reverse=True
        )
        for factor, contribution in sorted_factors[:3]:
            print(f"     - {factor}: {contribution:.3f}")
        
        print(f"   Recommended Inspections:")
        for inspection in mode_risk.recommended_inspections[:3]:
            print(f"     - {inspection}")
        
        print(f"   Recommended Actions:")
        for action in mode_risk.recommended_actions[:3]:
            print(f"     - {action.replace('_', ' ').title()}")
    
    print()
    print("=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

