#!/usr/bin/env python3
"""
Example: Inspection Realism Modeling

Demonstrates realistic inspection modeling with:
- Missed defects
- Inspector bias and skill levels
- Probability of detection by method
- Optimal inspection intervals
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from polerisk.pole_health.inspection_realism import (
    InspectionRealismModel,
    InspectorProfile,
    InspectionMethod,
    DefectType,
    Defect,
)
from datetime import datetime
import numpy as np


def create_actual_defects() -> list[Defect]:
    """Create a set of actual defects that exist on a pole."""
    return [
        Defect(
            defect_type=DefectType.GROUNDLINE_DECAY,
            severity=0.75,
            location="groundline",
            depth_cm=15.0,
        ),
        Defect(
            defect_type=DefectType.CRACK,
            severity=0.50,
            location="mid_pole",
            depth_cm=5.0,
        ),
        Defect(
            defect_type=DefectType.SURFACE_DECAY,
            severity=0.30,
            location="groundline",
            depth_cm=2.0,
        ),
    ]


def create_inspector_profiles() -> dict[str, InspectorProfile]:
    """Create different inspector profiles for comparison."""
    return {
        'novice': InspectorProfile(
            inspector_id="INS-001",
            experience_years=1.0,
            certification_level="certified",
            detection_bias=0.85,  # Under-detection tendency
            severity_bias=0.90,  # Under-estimate severity
            visual_skill=0.60,
            equipment_skill=0.50,
            documentation_skill=0.65,
            consistency=0.70,
        ),
        'experienced': InspectorProfile(
            inspector_id="INS-002",
            experience_years=8.0,
            certification_level="senior",
            detection_bias=1.0,  # No bias
            severity_bias=1.0,
            visual_skill=0.85,
            equipment_skill=0.80,
            documentation_skill=0.90,
            consistency=0.90,
        ),
        'expert': InspectorProfile(
            inspector_id="INS-003",
            experience_years=20.0,
            certification_level="expert",
            specialization="wood",
            detection_bias=1.05,  # Slight over-detection
            severity_bias=1.02,
            visual_skill=0.95,
            equipment_skill=0.90,
            documentation_skill=0.95,
            consistency=0.95,
        ),
    }


def main():
    """Run inspection realism demonstration."""
    print("=" * 70)
    print("Inspection Realism Modeling Demonstration")
    print("=" * 70)
    print()
    
    # Initialize model
    model = InspectionRealismModel()
    
    # Create actual defects
    actual_defects = create_actual_defects()
    print(f"Simulating inspection with {len(actual_defects)} actual defects:")
    for defect in actual_defects:
        print(f"  - {defect.defect_type.value}: severity={defect.severity:.2f}, "
              f"location={defect.location}, depth={defect.depth_cm}cm")
    print()
    
    # Test different inspector skill levels
    inspectors = create_inspector_profiles()
    methods = [
        InspectionMethod.VISUAL,
        InspectionMethod.SOUNDING,
        InspectionMethod.MOISTURE_METER,
    ]
    
    print("=" * 70)
    print("COMPARING INSPECTOR PERFORMANCE")
    print("=" * 70)
    print()
    
    results_by_inspector = {}
    
    for inspector_type, inspector in inspectors.items():
        print(f"\n{inspector_type.upper()} Inspector ({inspector.experience_years} years)")
        print("-" * 70)
        
        # Simulate inspection
        result = model.simulate_inspection(
            pole_id="POL-001",
            actual_defects=actual_defects,
            inspector=inspector,
            methods=methods,
            environmental_conditions={
                'poor_lighting': False,
                'rain': False,
                'vegetation_obstruction': True,  # Some obstruction
            }
        )
        
        results_by_inspector[inspector_type] = result
        
        # Display results
        print(f"Methods used: {[m.value for m in result.methods_used]}")
        print(f"Defects detected: {len([d for d in result.defects_detected if d.actual_present])}")
        print(f"Defects missed: {len(result.defects_missed)}")
        print(f"False positives: {len([d for d in result.defects_detected if not d.actual_present])}")
        print(f"Detection rate: {result.detection_rate:.1%}")
        print(f"False positive rate: {result.false_positive_rate:.1%}")
        print(f"Inspection confidence: {result.inspection_confidence:.1%}")
        print(f"Time spent: {result.time_spent_minutes:.1f} minutes")
        
        print(f"\nDetected defects:")
        for defect in result.defects_detected:
            if defect.actual_present:
                print(f"  + {defect.defect_type.value} (severity={defect.severity:.2f}) "
                      f"via {defect.detection_method.value if defect.detection_method else 'unknown'}")
        
        if result.defects_missed:
            print(f"\nMissed defects:")
            for defect in result.defects_missed:
                print(f"  - {defect.defect_type.value} (severity={defect.severity:.2f})")
    
    # Compare methods
    print("\n" + "=" * 70)
    print("COMPARING INSPECTION METHODS")
    print("=" * 70)
    print()
    
    method_combinations = [
        ([InspectionMethod.VISUAL], "Visual only"),
        ([InspectionMethod.VISUAL, InspectionMethod.SOUNDING], "Visual + Sounding"),
        ([InspectionMethod.VISUAL, InspectionMethod.SOUNDING, InspectionMethod.RESISTOGRAPH], 
         "Visual + Sounding + Resistograph"),
    ]
    
    inspector = inspectors['experienced']
    
    for methods_combo, description in method_combinations:
        result = model.simulate_inspection(
            pole_id="POL-001",
            actual_defects=actual_defects,
            inspector=inspector,
            methods=methods_combo,
        )
        
        total_cost = sum([
            model.method_capabilities[m].cost_per_pole 
            for m in methods_combo
        ])
        
        print(f"{description}:")
        print(f"  Detection rate: {result.detection_rate:.1%}")
        print(f"  Cost: ${total_cost:.0f}")
        print(f"  Time: {result.time_spent_minutes:.0f} minutes")
        print()
    
    # Optimal inspection interval
    print("=" * 70)
    print("OPTIMAL INSPECTION INTERVAL CALCULATION")
    print("=" * 70)
    print()
    
    risk_scores = [0.2, 0.5, 0.8]
    growth_rates = [0.05, 0.10, 0.20]
    
    print("Risk Score | Growth Rate | Optimal Interval")
    print("-" * 70)
    for risk in risk_scores:
        for growth in growth_rates:
            interval_days = model.calculate_optimal_inspection_interval(risk, growth)
            interval_str = f"{interval_days} days" if interval_days < 365 else f"{interval_days // 365} year(s)"
            print(f"   {risk:.1f}    |     {growth:.2f}    |   {interval_str}")
    
    # Method recommendations
    print("\n" + "=" * 70)
    print("INSPECTION METHOD RECOMMENDATIONS")
    print("=" * 70)
    print()
    
    defect_types = [DefectType.GROUNDLINE_DECAY, DefectType.INTERNAL_DECAY]
    budgets = [200.0, 500.0, 1000.0]
    
    for budget in budgets:
        recommended = model.recommend_inspection_methods(defect_types, budget)
        total_cost = sum([
            model.method_capabilities[m].cost_per_pole 
            for m in recommended
        ])
        print(f"Budget: ${budget:.0f}")
        print(f"  Recommended methods: {[m.value for m in recommended]}")
        print(f"  Total cost: ${total_cost:.0f}")
        print()
    
    print("=" * 70)
    print("Demonstration complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

