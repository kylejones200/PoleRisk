#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Launch script for the Utility Pole Health Dashboard.
"""

import os
import sys
import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_data_files():
    """Check if required data files exist."""
    required_files = [
        'Output/pole_health_assessment.csv',
        'Output/maintenance_schedule.csv'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        logger.warning("Missing required data files:")
        for file in missing_files:
            logger.warning(f"  - {file}")
        
        logger.info("Please run the pole assessment first:")
        logger.info("  python main.py --create-sample-data")
        logger.info("  python main.py --poles Input/sample_poles.csv --soil Input/sample_soil_data.csv")
        return False
    
    return True


def launch_dashboard():
    """Launch the Streamlit dashboard."""
    logger.info(" Launching Utility Pole Health Dashboard...")
    
    # Check for required data
    if not check_data_files():
        return
    
    try:
        # Launch Streamlit app
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'dashboard_app.py',
            '--server.port', '8501',
            '--server.address', 'localhost',
            '--browser.gatherUsageStats', 'false'
        ])
        
    except KeyboardInterrupt:
        logger.info("\n Dashboard stopped by user")
    except Exception as e:
        logger.error(f"Error launching dashboard: {e}")


if __name__ == "__main__":
    launch_dashboard()
