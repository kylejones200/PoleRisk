#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate comprehensive visualizations for utility pole health assessment.
"""

import os
import sys
import argparse
import pandas as pd
import logging
from pathlib import Path

from soilmoisture.visualization.pole_health_viz import PoleHealthVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_assessment_data(assessment_file: str, schedule_file: str, 
                        soil_file: str = None) -> tuple:
    """Load assessment data for visualization."""
    try:
        # Load assessment results
        assessment_df = pd.read_csv(assessment_file)
        logger.info(f"Loaded {len(assessment_df)} pole assessments from {assessment_file}")
        
        # Load maintenance schedule
        schedule_df = pd.read_csv(schedule_file)
        logger.info(f"Loaded {len(schedule_df)} maintenance items from {schedule_file}")
        
        # Load soil history if available
        soil_history = None
        if soil_file and os.path.exists(soil_file):
            soil_history = pd.read_csv(soil_file)
            logger.info(f"Loaded {len(soil_history)} soil samples from {soil_file}")
        
        return assessment_df, schedule_df, soil_history
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None, None, None


def generate_all_visualizations(assessment_file: str, schedule_file: str,
                               soil_file: str = None, output_dir: str = 'Analysis'):
    """Generate complete visualization suite."""
    logger.info("Starting visualization generation...")
    
    # Load data
    assessment_df, schedule_df, soil_history = load_assessment_data(
        assessment_file, schedule_file, soil_file
    )
    
    if assessment_df is None or schedule_df is None:
        logger.error("Failed to load required data files")
        return
    
    # Create visualizer
    visualizer = PoleHealthVisualizer()
    
    # Generate all visualizations
    generated_files = visualizer.generate_visual_report(
        assessment_df, schedule_df, soil_history, output_dir
    )
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("VISUALIZATION GENERATION COMPLETE")
    logger.info("="*60)
    
    for viz_type, file_path in generated_files.items():
        if file_path:
            logger.info(f"{viz_type.title()}: {file_path}")
    
    logger.info(f"\nAll visualizations saved to: {output_dir}")
    
    # Print what to do next
    if 'map' in generated_files:
        logger.info(f"\nOpen the interactive map: {generated_files['map']}")
    
    if 'dashboard' in generated_files:
        logger.info(f"View the dashboard image: {generated_files['dashboard']}")


def main():
    """Main entry point for visualization generation."""
    parser = argparse.ArgumentParser(
        description="Generate Utility Pole Health Visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_pole_health.py
  python visualize_pole_health.py --output Visuals/
  python visualize_pole_health.py --assessment custom_assessment.csv --schedule custom_schedule.csv
        """
    )
    
    parser.add_argument('--assessment', type=str, 
                       default='Output/pole_health_assessment.csv',
                       help='Path to pole health assessment CSV file')
    parser.add_argument('--schedule', type=str,
                       default='Output/maintenance_schedule.csv', 
                       help='Path to maintenance schedule CSV file')
    parser.add_argument('--soil', type=str,
                       help='Path to soil history CSV file (optional)')
    parser.add_argument('--output', type=str, default='Analysis',
                       help='Output directory for visualizations (default: Analysis)')
    
    args = parser.parse_args()
    
    try:
        # Check if required files exist
        if not os.path.exists(args.assessment):
            logger.error(f"Assessment file not found: {args.assessment}")
            logger.info("Run the main pole assessment first: python main.py --create-sample-data && python main.py --poles Input/sample_poles.csv --soil Input/sample_soil_data.csv")
            return
        
        if not os.path.exists(args.schedule):
            logger.error(f"Schedule file not found: {args.schedule}")
            logger.info("Run the main pole assessment first: python main.py --create-sample-data && python main.py --poles Input/sample_poles.csv --soil Input/sample_soil_data.csv")
            return
        
        # Generate visualizations
        generate_all_visualizations(
            args.assessment, 
            args.schedule, 
            args.soil, 
            args.output
        )
        
    except Exception as e:
        logger.exception("Error during visualization generation")


if __name__ == "__main__":
    main()
