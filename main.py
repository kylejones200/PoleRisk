#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility Pole Soil Assessment System

Main entry point for assessing utility pole health based on soil conditions.
"""

import os
import sys
import argparse
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path

from polerisk.pole_health.pole_data import PoleInfo, SoilSample, PoleDatabase
from polerisk.pole_health.assessment import PoleHealthAssessment
from polerisk.pole_health.risk_scoring import PoleRiskScorer, MaintenanceScheduler
from polerisk.common import DataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Data loading functions are now handled by the unified DataLoader
# These wrapper functions maintain backward compatibility
def load_pole_data(poles_file: str) -> list:
    """Load pole data from CSV file using unified DataLoader."""
    try:
        return DataLoader.load_pole_data(poles_file)
    except Exception as e:
        logger.error(f"Error loading pole data: {e}")
        return []


def load_soil_data(soil_file: str) -> list:
    """Load soil sample data from CSV file using unified DataLoader."""
    try:
        return DataLoader.load_soil_data(soil_file)
    except Exception as e:
        logger.error(f"Error loading soil data: {e}")
        return []


def assess_pole_fleet(poles: list, soil_samples: list, output_dir: str = "Output"):
    """Assess health of entire pole fleet."""
    logger.info("Starting pole fleet assessment...")
    
    # Initialize assessment system
    assessor = PoleHealthAssessment()
    risk_scorer = PoleRiskScorer()
    scheduler = MaintenanceScheduler()
    
    # Group soil samples by pole
    samples_by_pole = {}
    for sample in soil_samples:
        if sample.pole_id not in samples_by_pole:
            samples_by_pole[sample.pole_id] = []
        samples_by_pole[sample.pole_id].append(sample)
    
    # Assess each pole
    assessment_results = []
    for pole in poles:
        pole_samples = samples_by_pole.get(pole.pole_id, [])
        
        if not pole_samples:
            logger.warning(f"No soil samples found for pole {pole.pole_id}")
            continue
        
        # Perform health assessment
        health_metrics = assessor.assess_pole_health(pole, pole_samples)
        
        # Calculate comprehensive risk scores
        risk_scores = risk_scorer.calculate_comprehensive_risk_score(pole, health_metrics)
        
        assessment_results.append((pole, health_metrics, risk_scores))
    
    logger.info(f"Completed assessment for {len(assessment_results)} poles")
    
    # Generate maintenance schedule
    poles_and_metrics = [(pole, metrics) for pole, metrics, _ in assessment_results]
    schedule_df = scheduler.create_maintenance_schedule(poles_and_metrics)
    
    # Generate maintenance report
    maintenance_report = scheduler.generate_maintenance_report(schedule_df)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed assessment results
    assessment_data = []
    for pole, metrics, risks in assessment_results:
        data = {
            'pole_id': pole.pole_id,
            'latitude': pole.latitude,
            'longitude': pole.longitude,
            'pole_type': pole.pole_type,
            'age_years': pole.age_years,
            'overall_health_score': metrics.overall_health_score,
            'soil_stability_score': metrics.soil_stability_score,
            'structural_risk_score': metrics.structural_risk_score,
            'moisture_risk': metrics.moisture_risk,
            'erosion_risk': metrics.erosion_risk,
            'chemical_corrosion_risk': metrics.chemical_corrosion_risk,
            'bearing_capacity_risk': metrics.bearing_capacity_risk,
            'maintenance_priority': metrics.maintenance_priority,
            'requires_immediate_attention': metrics.requires_immediate_attention,
            'confidence_level': metrics.confidence_level,
            'safety_risk': risks.get('safety_risk', 0),
            'reliability_risk': risks.get('reliability_risk', 0),
            'financial_risk': risks.get('financial_risk', 0),
            'composite_risk': risks.get('composite_risk', 0)
        }
        assessment_data.append(data)
    
    assessment_df = pd.DataFrame(assessment_data)
    assessment_file = os.path.join(output_dir, "pole_health_assessment.csv")
    assessment_df.to_csv(assessment_file, index=False)
    logger.info(f"Saved detailed assessment to {assessment_file}")
    
    # Save maintenance schedule
    schedule_file = os.path.join(output_dir, "maintenance_schedule.csv")
    schedule_df.to_csv(schedule_file, index=False)
    logger.info(f"Saved maintenance schedule to {schedule_file}")
    
    # Save summary report
    summary_file = os.path.join(output_dir, "maintenance_report.txt")
    with open(summary_file, 'w') as f:
        f.write("UTILITY POLE MAINTENANCE REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Assessment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("SUMMARY\n")
        f.write("-" * 20 + "\n")
        summary = maintenance_report['summary']
        f.write(f"Total Poles Assessed: {summary['total_poles']}\n")
        f.write(f"Poles Requiring Urgent Attention: {summary['urgent_poles']}\n")
        f.write(f"Urgent Poles Percentage: {summary['urgent_percentage']:.1f}%\n")
        f.write(f"Total Estimated Maintenance Cost: ${summary['total_estimated_cost']:,.2f}\n")
        f.write(f"Average Cost Per Pole: ${summary['average_cost_per_pole']:,.2f}\n\n")
        
        f.write("PRIORITY BREAKDOWN\n")
        f.write("-" * 20 + "\n")
        priority_counts = maintenance_report['by_priority']['counts']
        priority_costs = maintenance_report['by_priority']['costs']
        
        for priority in ['critical', 'high', 'medium', 'low']:
            count = priority_counts.get(priority, 0)
            cost = priority_costs.get(priority, 0)
            f.write(f"{priority.capitalize()}: {count} poles, ${cost:,.2f}\n")
    
    logger.info(f"Saved summary report to {summary_file}")
    
    return assessment_results, schedule_df, maintenance_report


def create_sample_data(output_dir: str = "Input"):
    """Create sample pole and soil data files for demonstration."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample pole data
    pole_data = [
        {
            'pole_id': 'P001',
            'latitude': 40.7128,
            'longitude': -74.0060,
            'pole_type': 'wood',
            'material': 'Southern Pine',
            'height_ft': 45,
            'install_date': '2010-05-15',
            'voltage_class': 'distribution',
            'structure_type': 'tangent',
            'treatment_type': 'CCA'
        },
        {
            'pole_id': 'P002',
            'latitude': 40.7580,
            'longitude': -73.9855,
            'pole_type': 'concrete',
            'material': 'Prestressed Concrete',
            'height_ft': 50,
            'install_date': '2005-08-22',
            'voltage_class': 'transmission',
            'structure_type': 'dead-end'
        },
        {
            'pole_id': 'P003',
            'latitude': 40.6782,
            'longitude': -73.9442,
            'pole_type': 'steel',
            'material': 'Galvanized Steel',
            'height_ft': 55,
            'install_date': '2015-03-10',
            'voltage_class': 'transmission',
            'structure_type': 'corner'
        }
    ]
    
    poles_df = pd.DataFrame(pole_data)
    poles_file = os.path.join(output_dir, "sample_poles.csv")
    poles_df.to_csv(poles_file, index=False)
    
    # Sample soil data
    soil_data = []
    import random
    
    for pole_id in ['P001', 'P002', 'P003']:
        # Generate multiple samples over time for each pole
        for i in range(3):
            base_date = datetime(2023, 1, 1)
            sample_date = base_date + pd.Timedelta(days=i*90)  # Quarterly samples
            
            soil_data.append({
                'pole_id': pole_id,
                'sample_date': sample_date.strftime('%Y-%m-%d'),
                'depth_inches': 12,
                'moisture_content': round(random.uniform(0.15, 0.40), 3),
                'ph': round(random.uniform(6.0, 8.5), 1),
                'bulk_density': round(random.uniform(1.2, 1.8), 2),
                'electrical_conductivity': round(random.uniform(0.5, 3.0), 2),
                'bearing_capacity': round(random.uniform(80, 300), 0),
                'soil_type': random.choice(['clay', 'sandy_loam', 'silt']),
                'data_quality': 'good'
            })
    
    soil_df = pd.DataFrame(soil_data)
    soil_file = os.path.join(output_dir, "sample_soil_data.csv")
    soil_df.to_csv(soil_file, index=False)
    
    logger.info(f"Created sample data files:")
    logger.info(f"  - Poles: {poles_file}")
    logger.info(f"  - Soil: {soil_file}")
    
    return poles_file, soil_file


def main():
    """Main entry point for utility pole assessment system."""
    parser = argparse.ArgumentParser(
        description="Utility Pole Soil Assessment System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --create-sample-data
  python main.py --poles Input/poles.csv --soil Input/soil_data.csv
  python main.py --poles Input/poles.csv --soil Input/soil_data.csv --output Results/
        """
    )
    
    parser.add_argument('--poles', type=str, help='Path to poles CSV file')
    parser.add_argument('--soil', type=str, help='Path to soil data CSV file')
    parser.add_argument('--output', type=str, default='Output', 
                       help='Output directory for results (default: Output)')
    parser.add_argument('--create-sample-data', action='store_true',
                       help='Create sample data files for demonstration')
    
    args = parser.parse_args()
    
    try:
        if args.create_sample_data:
            logger.info("Creating sample data files...")
            poles_file, soil_file = create_sample_data()
            logger.info("Sample data created successfully!")
            logger.info(f"Run: python {sys.argv[0]} --poles {poles_file} --soil {soil_file}")
            return
        
        if not args.poles or not args.soil:
            logger.error("Both --poles and --soil arguments are required")
            logger.info("Use --create-sample-data to generate sample files first")
            parser.print_help()
            return
        
        # Load data
        poles = load_pole_data(args.poles)
        soil_samples = load_soil_data(args.soil)
        
        if not poles:
            logger.error("No pole data loaded. Check your poles file.")
            return
        
        if not soil_samples:
            logger.error("No soil data loaded. Check your soil data file.")
            return
        
        # Perform assessment
        results, schedule, report = assess_pole_fleet(poles, soil_samples, args.output)
        
        # Print summary
        logger.info("\nASSESSMENT COMPLETE")
        logger.info("=" * 50)
        logger.info(f"Assessed {len(results)} poles")
        logger.info(f"Results saved to: {args.output}")
        
        # Show immediate attention poles
        urgent_poles = schedule[schedule['requires_immediate_attention'] == True]
        if len(urgent_poles) > 0:
            logger.warning(f"\n{len(urgent_poles)} poles require immediate attention:")
            for _, pole in urgent_poles.iterrows():
                logger.warning(f"  - {pole['pole_id']}: {pole['recommended_actions']}")
        
        logger.info("\nCheck the output directory for detailed results:")
        logger.info(f"  - pole_health_assessment.csv: Detailed health scores")
        logger.info(f"  - maintenance_schedule.csv: Prioritized maintenance schedule")
        logger.info(f"  - maintenance_report.txt: Executive summary")
        
    except Exception as e:
        logger.exception("Error during assessment")


if __name__ == "__main__":
    main()
