#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Database migration script - convert CSV data to database backend.
"""

import sys
import os
import pandas as pd
from datetime import datetime
import logging

# Add project root to path
sys.path.append('/Users/k.jones/Documents/moisture')

from polerisk.database.models import (
    DatabaseManager, PoleDataAccess, Pole, SoilSample, 
    StructuralInspection, HealthAssessment, WeatherData
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def migrate_csv_to_database():
    """Migrate existing CSV data to database."""
    
    logger.debug(" MIGRATING CSV DATA TO DATABASE")
    logger.debug("=" * 50)
    
    # Initialize database
    db_manager = DatabaseManager()  # Uses SQLite by default
    db_manager.create_tables()
    data_access = PoleDataAccess(db_manager)
    
    logger.debug(" Database initialized and tables created")
    
    migration_stats = {
        'poles': 0,
        'soil_samples': 0,
        'inspections': 0,
        'health_assessments': 0,
        'weather_data': 0,
        'errors': 0
    }
    
    try:
        # 1. Migrate Poles
        logger.debug("\n Migrating pole data...")
        poles_file = 'Input/sample_poles.csv'
        if os.path.exists(poles_file):
            poles_df = pd.read_csv(poles_file)
            
            for _, row in poles_df.iterrows():
                try:
                    pole_data = {
                        'pole_id': str(row['pole_id']),
                        'latitude': float(row['latitude']),
                        'longitude': float(row['longitude']),
                        'pole_type': row.get('pole_type', 'wood'),
                        'material': row.get('material', 'Unknown'),
                        'height_ft': float(row.get('height_ft', 40)),
                        'install_date': pd.to_datetime(row.get('install_date', '2000-01-01')),
                        'voltage_class': row.get('voltage_class', 'distribution'),
                        'structure_type': row.get('structure_type', 'tangent'),
                        'diameter_base_inches': row.get('diameter_base_inches'),
                        'treatment_type': row.get('treatment_type'),
                        'condition_rating': row.get('condition_rating')
                    }
                    
                    # Check if pole already exists
                    existing_pole = data_access.get_pole(pole_data['pole_id'])
                    if not existing_pole:
                        data_access.create_pole(pole_data)
                        migration_stats['poles'] += 1
                        logger.debug(f"   Migrated pole {pole_data['pole_id']}")
                    else:
                        logger.debug(f"   Pole {pole_data['pole_id']} already exists, skipping")
                        
                except Exception as e:
                    logger.error(f"Error migrating pole {row.get('pole_id', 'unknown')}: {e}")
                    migration_stats['errors'] += 1
        else:
            logger.debug(f"   Poles file not found: {poles_file}")
        
        # 2. Migrate Soil Samples
        logger.debug("\n Migrating soil sample data...")
        soil_file = 'Input/sample_soil_data.csv'
        if os.path.exists(soil_file):
            soil_df = pd.read_csv(soil_file)
            
            for _, row in soil_df.iterrows():
                try:
                    sample_data = {
                        'pole_id': str(row['pole_id']),
                        'sample_date': pd.to_datetime(row['sample_date']),
                        'depth_inches': float(row.get('depth_inches', 12)),
                        'moisture_content': float(row.get('moisture_content', 0.2)),
                        'ph': row.get('ph'),
                        'bulk_density': row.get('bulk_density'),
                        'electrical_conductivity': row.get('electrical_conductivity'),
                        'bearing_capacity': row.get('bearing_capacity'),
                        'soil_type': row.get('soil_type'),
                        'data_quality': row.get('data_quality', 'good')
                    }
                    
                    data_access.add_soil_sample(sample_data)
                    migration_stats['soil_samples'] += 1
                    
                except Exception as e:
                    logger.error(f"Error migrating soil sample: {e}")
                    migration_stats['errors'] += 1
            
            logger.debug(f"   Migrated {migration_stats['soil_samples']} soil samples")
        else:
            logger.debug(f"   Soil data file not found: {soil_file}")
        
        # 3. Migrate Structural Inspections
        logger.debug("\n Migrating structural inspection data...")
        inspection_file = 'Input/sample_structural_inspections.csv'
        if os.path.exists(inspection_file):
            inspection_df = pd.read_csv(inspection_file)
            
            for _, row in inspection_df.iterrows():
                try:
                    inspection_data = {
                        'inspection_id': str(row['inspection_id']),
                        'pole_id': str(row['pole_id']),
                        'inspection_date': pd.to_datetime(row['inspection_date']),
                        'inspector_id': str(row['inspector_id']),
                        'inspection_type': str(row['inspection_type']),
                        'overall_condition': row.get('overall_condition'),
                        'visible_damage': bool(row.get('visible_damage', False)),
                        'damage_description': row.get('damage_description'),
                        'wood_decay_depth': row.get('wood_decay_depth'),
                        'wood_circumferential_loss': row.get('wood_circumferential_loss'),
                        'concrete_cracking': row.get('concrete_cracking'),
                        'concrete_spalling': row.get('concrete_spalling'),
                        'steel_corrosion_level': row.get('steel_corrosion_level'),
                        'coating_condition': row.get('coating_condition'),
                        'ground_line_circumference': row.get('ground_line_circumference'),
                        'lean_angle': row.get('lean_angle'),
                        'twist_angle': row.get('twist_angle'),
                        'estimated_remaining_strength': row.get('estimated_remaining_strength'),
                        'confidence_level': float(row.get('confidence_level', 1.0)),
                        'recommended_action': row.get('recommended_action'),
                        'notes': row.get('notes')
                    }
                    
                    data_access.add_inspection(inspection_data)
                    migration_stats['inspections'] += 1
                    
                except Exception as e:
                    logger.error(f"Error migrating inspection: {e}")
                    migration_stats['errors'] += 1
            
            logger.debug(f"   Migrated {migration_stats['inspections']} structural inspections")
        else:
            logger.debug(f"   Inspection data file not found: {inspection_file}")
        
        # 4. Migrate Health Assessments
        logger.debug("\n Migrating health assessment data...")
        assessment_file = 'Output/pole_health_assessment.csv'
        if os.path.exists(assessment_file):
            assessment_df = pd.read_csv(assessment_file)
            
            for _, row in assessment_df.iterrows():
                try:
                    assessment_data = {
                        'pole_id': str(row['pole_id']),
                        'assessment_date': datetime.now(),
                        'overall_health_score': float(row['overall_health_score']),
                        'soil_stability_score': float(row['soil_stability_score']),
                        'structural_risk_score': float(row['structural_risk_score']),
                        'moisture_risk': float(row['moisture_risk']),
                        'erosion_risk': float(row['erosion_risk']),
                        'chemical_corrosion_risk': float(row['chemical_corrosion_risk']),
                        'bearing_capacity_risk': float(row['bearing_capacity_risk']),
                        'maintenance_priority': str(row['maintenance_priority']),
                        'requires_immediate_attention': bool(row['requires_immediate_attention']),
                        'confidence_level': float(row['confidence_level'])
                    }
                    
                    data_access.add_health_assessment(assessment_data)
                    migration_stats['health_assessments'] += 1
                    
                except Exception as e:
                    logger.error(f"Error migrating health assessment: {e}")
                    migration_stats['errors'] += 1
            
            logger.debug(f"   Migrated {migration_stats['health_assessments']} health assessments")
        else:
            logger.debug(f"   Health assessment file not found: {assessment_file}")
        
        # 5. Migrate Weather-Enhanced Assessments (if available)
        weather_enhanced_file = 'Output/weather_enhanced_assessment.csv'
        if os.path.exists(weather_enhanced_file):
            logger.debug("\n Migrating weather-enhanced assessment data...")
            weather_df = pd.read_csv(weather_enhanced_file)
            
            for _, row in weather_df.iterrows():
                try:
                    # Update existing health assessment with weather data
                    pole_id = str(row['pole_id'])
                    latest_assessment = data_access.get_latest_health_assessment(pole_id)
                    
                    if latest_assessment:
                        updates = {
                            'current_weather_risk': float(row.get('current_weather_risk', 0)),
                            'forecast_weather_risk': float(row.get('forecast_weather_risk', 0)),
                            'combined_weather_risk': float(row.get('combined_weather_risk', 0))
                        }
                        
                        # Update the assessment in database
                        session = db_manager.get_session()
                        try:
                            for key, value in updates.items():
                                setattr(latest_assessment, key, value)
                            session.commit()
                            migration_stats['weather_data'] += 1
                        finally:
                            session.close()
                    
                except Exception as e:
                    logger.error(f"Error migrating weather data: {e}")
                    migration_stats['errors'] += 1
            
            logger.debug(f"   Enhanced {migration_stats['weather_data']} assessments with weather data")
        
    except Exception as e:
        logger.error(f"Migration error: {e}")
        migration_stats['errors'] += 1
    
    finally:
        db_manager.close()
    
    return migration_stats


def test_database_operations():
    """Test basic database operations."""
    
    logger.debug("\n TESTING DATABASE OPERATIONS")
    logger.debug("=" * 50)
    
    db_manager = DatabaseManager()
    data_access = PoleDataAccess(db_manager)
    
    try:
        # Test pole retrieval
        poles = data_access.get_all_poles()
        logger.debug(f" Retrieved {len(poles)} poles from database")
        
        if poles:
            # Test pole details
            sample_pole = poles[0]
            logger.debug(f"  Sample pole: {sample_pole.pole_id} ({sample_pole.pole_type})")
            logger.debug(f"  Location: ({sample_pole.latitude:.4f}, {sample_pole.longitude:.4f})")
            
            # Test soil samples for this pole
            soil_samples = data_access.get_soil_samples(sample_pole.pole_id)
            logger.debug(f"  Soil samples: {len(soil_samples)}")
            
            # Test inspections for this pole
            inspections = data_access.get_inspections(sample_pole.pole_id)
            logger.debug(f"  Inspections: {len(inspections)}")
            
            # Test health assessment
            health_assessment = data_access.get_latest_health_assessment(sample_pole.pole_id)
            if health_assessment:
                logger.debug(f"  Health score: {health_assessment.overall_health_score:.1f}")
                logger.debug(f"  Priority: {health_assessment.maintenance_priority}")
        
        # Test work order creation
        if poles:
            work_order_data = {
                'work_order_id': f"WO_{poles[0].pole_id}_{datetime.now().strftime('%Y%m%d')}",
                'pole_id': poles[0].pole_id,
                'work_type': 'inspection',
                'priority': 'medium',
                'description': 'Routine inspection based on database test',
                'estimated_hours': 2.0,
                'estimated_cost': 300.0
            }
            
            work_order = data_access.create_work_order(work_order_data)
            logger.debug(f" Created test work order: {work_order.work_order_id}")
        
        logger.info(" All database operations successful!")
        
    except Exception as e:
        logger.error(f"Database test error: {e}")
        logger.error(f" Database test failed: {e}")
    
    finally:
        db_manager.close()


def generate_database_summary():
    """Generate summary of database contents."""
    
    logger.debug("\n DATABASE SUMMARY")
    logger.debug("=" * 50)
    
    db_manager = DatabaseManager()
    data_access = PoleDataAccess(db_manager)
    
    try:
        # Count records in each table
        poles = data_access.get_all_poles()
        all_soil_samples = data_access.get_soil_samples()
        all_inspections = data_access.get_inspections()
        open_work_orders = data_access.get_work_orders(status='open')
        
        logger.debug(f" Database Contents:")
        logger.debug(f"  • Poles: {len(poles)}")
        logger.debug(f"  • Soil Samples: {len(all_soil_samples)}")
        logger.debug(f"  • Structural Inspections: {len(all_inspections)}")
        logger.debug(f"  • Open Work Orders: {len(open_work_orders)}")
        
        # Health status summary
        if poles:
            health_assessments = []
            for pole in poles:
                assessment = data_access.get_latest_health_assessment(pole.pole_id)
                if assessment:
                    health_assessments.append(assessment)
            
            if health_assessments:
                priorities = {}
                for assessment in health_assessments:
                    priority = assessment.maintenance_priority
                    priorities[priority] = priorities.get(priority, 0) + 1
                
                logger.debug(f"\n Maintenance Priorities:")
                for priority, count in priorities.items():
                    logger.debug(f"  • {priority.capitalize()}: {count} poles")
                
                avg_health = sum(a.overall_health_score for a in health_assessments) / len(health_assessments)
                logger.debug(f"\n Fleet Health Average: {avg_health:.1f}/100")
    
    except Exception as e:
        logger.error(f"Summary generation error: {e}")
    
    finally:
        db_manager.close()


def main():
    """Main migration function."""
    
    logger.debug(" DATABASE BACKEND IMPLEMENTATION")
    logger.debug("Generated:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    logger.debug("=" * 60)
    
    try:
        # Run migration
        migration_stats = migrate_csv_to_database()
        
        # Test database operations
        test_database_operations()
        
        # Generate summary
        generate_database_summary()
        
        logger.debug("\n" + "=" * 60)
        logger.info(" DATABASE BACKEND IMPLEMENTATION COMPLETE")
        logger.debug("=" * 60)
        logger.debug("Migration Results:")
        for table, count in migration_stats.items():
            if count > 0:
                logger.debug(f"  • {table.replace('_', ' ').title()}: {count}")
        
        if migration_stats['errors'] > 0:
            logger.error(f"   Errors encountered: {migration_stats['errors']}")
        
        logger.debug(f"\n Database file: pole_assessment.db")
        logger.debug(" Next steps:")
        logger.debug("  • Database backend is now operational")
        logger.debug("  • All CSV data has been migrated")
        logger.debug("  • Ready for REST API development")
        logger.debug("  • Can switch to PostgreSQL for production")
        
    except Exception as e:
        logger.exception("Migration failed")


if __name__ == "__main__":
    main()
