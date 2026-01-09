"""
REST API endpoints for utility pole health assessment system.
"""

from fastapi import FastAPI, HTTPException, Depends, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, validator
from datetime import datetime, date
import logging
import os
import pandas as pd
import io

# Import database models and data access
from polerisk.database import DatabaseManager, PoleDataAccess
from polerisk.pole_health.assessment import PoleHealthAssessment
from polerisk.pole_health.structural_assessment import EnhancedPoleHealthAssessment
from polerisk.weather import OpenWeatherMapProvider, WeatherRiskAssessment

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Utility Pole Health Assessment API",
    description="REST API for managing utility pole health assessments, maintenance scheduling, and risk analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database dependency
def get_database():
    """Get database connection."""
    db_manager = DatabaseManager()
    try:
        yield PoleDataAccess(db_manager)
    finally:
        db_manager.close()

# Pydantic models for request/response
class PoleCreate(BaseModel):
    pole_id: str
    latitude: float
    longitude: float
    pole_type: str
    material: str
    height_ft: float
    install_date: date
    voltage_class: str
    structure_type: str
    diameter_base_inches: Optional[float] = None
    treatment_type: Optional[str] = None
    condition_rating: Optional[str] = None

class PoleResponse(BaseModel):
    id: int
    pole_id: str
    latitude: float
    longitude: float
    pole_type: str
    material: str
    height_ft: float
    install_date: datetime
    voltage_class: str
    structure_type: str
    condition_rating: Optional[str] = None
    created_at: datetime
    updated_at: datetime

class SoilSampleCreate(BaseModel):
    pole_id: str
    sample_date: date
    depth_inches: float
    moisture_content: float
    ph: Optional[float] = None
    bulk_density: Optional[float] = None
    electrical_conductivity: Optional[float] = None
    bearing_capacity: Optional[float] = None
    soil_type: Optional[str] = None

class SoilSampleResponse(BaseModel):
    id: int
    pole_id: str
    sample_date: datetime
    depth_inches: float
    moisture_content: float
    ph: Optional[float] = None
    bulk_density: Optional[float] = None
    electrical_conductivity: Optional[float] = None
    bearing_capacity: Optional[float] = None
    soil_type: Optional[str] = None
    created_at: datetime

class HealthAssessmentResponse(BaseModel):
    id: int
    pole_id: str
    assessment_date: datetime
    overall_health_score: float
    soil_stability_score: float
    structural_risk_score: float
    moisture_risk: float
    erosion_risk: float
    chemical_corrosion_risk: float
    bearing_capacity_risk: float
    maintenance_priority: str
    requires_immediate_attention: bool
    confidence_level: float

class WorkOrderCreate(BaseModel):
    pole_id: str
    work_type: str
    priority: str
    description: str
    estimated_hours: Optional[float] = None
    estimated_cost: Optional[float] = None
    scheduled_date: Optional[date] = None

class WorkOrderResponse(BaseModel):
    id: int
    work_order_id: str
    pole_id: str
    work_type: str
    priority: str
    description: str
    status: str
    created_date: datetime
    scheduled_date: Optional[datetime] = None
    estimated_cost: Optional[float] = None

# API Routes

@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "message": "Utility Pole Health Assessment API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "version": "1.0.0"
    }

# Pole Management Endpoints

@app.get("/poles", response_model=List[PoleResponse])
async def get_poles(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, le=1000),
    pole_type: Optional[str] = Query(None),
    db: PoleDataAccess = Depends(get_database)
):
    """Get all poles with optional filtering."""
    try:
        poles = db.get_all_poles()
        
        # Apply filters
        if pole_type:
            poles = [p for p in poles if p.pole_type == pole_type]
        
        # Apply pagination
        poles = poles[skip:skip + limit]
        
        return poles
    except Exception as e:
        logger.error(f"Error retrieving poles: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/poles/{pole_id}", response_model=PoleResponse)
async def get_pole(pole_id: str, db: PoleDataAccess = Depends(get_database)):
    """Get a specific pole by ID."""
    try:
        pole = db.get_pole(pole_id)
        if not pole:
            raise HTTPException(status_code=404, detail=f"Pole {pole_id} not found")
        return pole
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving pole {pole_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/poles", response_model=PoleResponse)
async def create_pole(pole_data: PoleCreate, db: PoleDataAccess = Depends(get_database)):
    """Create a new pole."""
    try:
        # Check if pole already exists
        existing_pole = db.get_pole(pole_data.pole_id)
        if existing_pole:
            raise HTTPException(status_code=400, detail=f"Pole {pole_data.pole_id} already exists")
        
        # Create pole
        pole_dict = pole_data.dict()
        pole = db.create_pole(pole_dict)
        return pole
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating pole: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.put("/poles/{pole_id}", response_model=PoleResponse)
async def update_pole(
    pole_id: str, 
    updates: Dict[str, Any], 
    db: PoleDataAccess = Depends(get_database)
):
    """Update a pole."""
    try:
        pole = db.update_pole(pole_id, updates)
        if not pole:
            raise HTTPException(status_code=404, detail=f"Pole {pole_id} not found")
        return pole
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating pole {pole_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Soil Sample Endpoints

@app.get("/poles/{pole_id}/soil-samples", response_model=List[SoilSampleResponse])
async def get_soil_samples(pole_id: str, db: PoleDataAccess = Depends(get_database)):
    """Get soil samples for a pole."""
    try:
        samples = db.get_soil_samples(pole_id)
        return samples
    except Exception as e:
        logger.error(f"Error retrieving soil samples for pole {pole_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/soil-samples", response_model=SoilSampleResponse)
async def create_soil_sample(
    sample_data: SoilSampleCreate, 
    db: PoleDataAccess = Depends(get_database)
):
    """Create a new soil sample."""
    try:
        # Verify pole exists
        pole = db.get_pole(sample_data.pole_id)
        if not pole:
            raise HTTPException(status_code=404, detail=f"Pole {sample_data.pole_id} not found")
        
        sample_dict = sample_data.dict()
        sample = db.add_soil_sample(sample_dict)
        return sample
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating soil sample: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Health Assessment Endpoints

@app.get("/poles/{pole_id}/health", response_model=HealthAssessmentResponse)
async def get_pole_health(pole_id: str, db: PoleDataAccess = Depends(get_database)):
    """Get latest health assessment for a pole."""
    try:
        assessment = db.get_latest_health_assessment(pole_id)
        if not assessment:
            raise HTTPException(status_code=404, detail=f"No health assessment found for pole {pole_id}")
        return assessment
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving health assessment for pole {pole_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/poles/{pole_id}/assess")
async def assess_pole_health(pole_id: str, db: PoleDataAccess = Depends(get_database)):
    """Trigger health assessment for a pole."""
    try:
        # Get pole data
        pole = db.get_pole(pole_id)
        if not pole:
            raise HTTPException(status_code=404, detail=f"Pole {pole_id} not found")
        
        # Get soil samples
        soil_samples = db.get_soil_samples(pole_id)
        if not soil_samples:
            raise HTTPException(status_code=400, detail=f"No soil samples available for pole {pole_id}")
        
        # Get structural inspections
        inspections = db.get_inspections(pole_id)
        
        # Convert database models to our assessment models
        from polerisk.pole_health.pole_data import PoleInfo, SoilSample
        
        pole_info = PoleInfo(
            pole_id=pole.pole_id,
            latitude=pole.latitude,
            longitude=pole.longitude,
            pole_type=pole.pole_type,
            material=pole.material,
            height_ft=pole.height_ft,
            install_date=pole.install_date,
            voltage_class=pole.voltage_class,
            structure_type=pole.structure_type,
            treatment_type=pole.treatment_type
        )
        
        soil_sample_objects = []
        for sample in soil_samples:
            soil_obj = SoilSample(
                pole_id=sample.pole_id,
                sample_date=sample.sample_date,
                depth_inches=sample.depth_inches,
                moisture_content=sample.moisture_content,
                ph=sample.ph,
                bulk_density=sample.bulk_density,
                electrical_conductivity=sample.electrical_conductivity,
                bearing_capacity=sample.bearing_capacity,
                soil_type=sample.soil_type,
                data_quality=sample.data_quality
            )
            soil_sample_objects.append(soil_obj)
        
        # Perform assessment
        if inspections:
            # Use enhanced assessment with structural data
            assessor = EnhancedPoleHealthAssessment()
            # Convert inspection data as needed
            health_metrics = assessor.assess_pole_with_structural_data(
                pole_info, soil_sample_objects, None, None
            )
        else:
            # Use basic soil assessment
            assessor = PoleHealthAssessment()
            health_metrics = assessor.assess_pole_health(pole_info, soil_sample_objects)
        
        # Save assessment to database
        assessment_data = {
            'pole_id': pole_id,
            'assessment_date': datetime.now(),
            'overall_health_score': health_metrics.overall_health_score,
            'soil_stability_score': health_metrics.soil_stability_score,
            'structural_risk_score': health_metrics.structural_risk_score,
            'moisture_risk': health_metrics.moisture_risk,
            'erosion_risk': health_metrics.erosion_risk,
            'chemical_corrosion_risk': health_metrics.chemical_corrosion_risk,
            'bearing_capacity_risk': health_metrics.bearing_capacity_risk,
            'maintenance_priority': health_metrics.maintenance_priority,
            'requires_immediate_attention': health_metrics.requires_immediate_attention,
            'confidence_level': health_metrics.confidence_level
        }
        
        assessment = db.add_health_assessment(assessment_data)
        
        return {
            "message": f"Health assessment completed for pole {pole_id}",
            "assessment_id": assessment.id,
            "health_score": assessment.overall_health_score,
            "priority": assessment.maintenance_priority,
            "requires_attention": assessment.requires_immediate_attention
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error assessing pole {pole_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Work Order Endpoints

@app.get("/work-orders", response_model=List[WorkOrderResponse])
async def get_work_orders(
    status: Optional[str] = Query(None),
    priority: Optional[str] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, le=1000),
    db: PoleDataAccess = Depends(get_database)
):
    """Get work orders with optional filtering."""
    try:
        work_orders = db.get_work_orders(status=status)
        
        # Apply priority filter
        if priority:
            work_orders = [wo for wo in work_orders if wo.priority == priority]
        
        # Apply pagination
        work_orders = work_orders[skip:skip + limit]
        
        return work_orders
    except Exception as e:
        logger.error(f"Error retrieving work orders: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/work-orders", response_model=WorkOrderResponse)
async def create_work_order(
    work_order_data: WorkOrderCreate, 
    db: PoleDataAccess = Depends(get_database)
):
    """Create a new work order."""
    try:
        # Verify pole exists
        pole = db.get_pole(work_order_data.pole_id)
        if not pole:
            raise HTTPException(status_code=404, detail=f"Pole {work_order_data.pole_id} not found")
        
        # Generate work order ID
        work_order_id = f"WO_{work_order_data.pole_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        work_order_dict = work_order_data.dict()
        work_order_dict['work_order_id'] = work_order_id
        
        work_order = db.create_work_order(work_order_dict)
        return work_order
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating work order: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Bulk Assessment Endpoint

@app.post("/assess-all")
async def assess_all_poles(db: PoleDataAccess = Depends(get_database)):
    """Trigger health assessment for all poles."""
    try:
        poles = db.get_all_poles()
        assessment_results = []
        
        for pole in poles:
            try:
                # Get soil samples
                soil_samples = db.get_soil_samples(pole.pole_id)
                if not soil_samples:
                    continue
                
                # Convert to assessment objects
                from polerisk.pole_health.pole_data import PoleInfo, SoilSample
                
                pole_info = PoleInfo(
                    pole_id=pole.pole_id,
                    latitude=pole.latitude,
                    longitude=pole.longitude,
                    pole_type=pole.pole_type,
                    material=pole.material,
                    height_ft=pole.height_ft,
                    install_date=pole.install_date,
                    voltage_class=pole.voltage_class,
                    structure_type=pole.structure_type,
                    treatment_type=pole.treatment_type
                )
                
                soil_sample_objects = []
                for sample in soil_samples:
                    soil_obj = SoilSample(
                        pole_id=sample.pole_id,
                        sample_date=sample.sample_date,
                        depth_inches=sample.depth_inches,
                        moisture_content=sample.moisture_content,
                        ph=sample.ph,
                        bulk_density=sample.bulk_density,
                        electrical_conductivity=sample.electrical_conductivity,
                        bearing_capacity=sample.bearing_capacity,
                        soil_type=sample.soil_type,
                        data_quality=sample.data_quality
                    )
                    soil_sample_objects.append(soil_obj)
                
                # Perform assessment
                assessor = PoleHealthAssessment()
                health_metrics = assessor.assess_pole_health(pole_info, soil_sample_objects)
                
                # Save to database
                assessment_data = {
                    'pole_id': pole.pole_id,
                    'assessment_date': datetime.now(),
                    'overall_health_score': health_metrics.overall_health_score,
                    'soil_stability_score': health_metrics.soil_stability_score,
                    'structural_risk_score': health_metrics.structural_risk_score,
                    'moisture_risk': health_metrics.moisture_risk,
                    'erosion_risk': health_metrics.erosion_risk,
                    'chemical_corrosion_risk': health_metrics.chemical_corrosion_risk,
                    'bearing_capacity_risk': health_metrics.bearing_capacity_risk,
                    'maintenance_priority': health_metrics.maintenance_priority,
                    'requires_immediate_attention': health_metrics.requires_immediate_attention,
                    'confidence_level': health_metrics.confidence_level
                }
                
                assessment = db.add_health_assessment(assessment_data)
                
                assessment_results.append({
                    'pole_id': pole.pole_id,
                    'health_score': assessment.overall_health_score,
                    'priority': assessment.maintenance_priority
                })
                
            except Exception as e:
                logger.error(f"Error assessing pole {pole.pole_id}: {e}")
                continue
        
        return {
            "message": f"Bulk assessment completed for {len(assessment_results)} poles",
            "results": assessment_results
        }
        
    except Exception as e:
        logger.error(f"Error in bulk assessment: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Data Upload Endpoint

@app.post("/upload/poles")
async def upload_poles_csv(
    file: UploadFile = File(...), 
    db: PoleDataAccess = Depends(get_database)
):
    """Upload poles data from CSV file."""
    try:
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        created_poles = []
        errors = []
        
        for _, row in df.iterrows():
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
                    'treatment_type': row.get('treatment_type')
                }
                
                # Check if pole exists
                existing_pole = db.get_pole(pole_data['pole_id'])
                if not existing_pole:
                    pole = db.create_pole(pole_data)
                    created_poles.append(pole_data['pole_id'])
                else:
                    errors.append(f"Pole {pole_data['pole_id']} already exists")
                    
            except Exception as e:
                errors.append(f"Error processing row: {e}")
        
        return {
            "message": f"Processed {len(df)} rows",
            "created": len(created_poles),
            "errors": len(errors),
            "created_poles": created_poles,
            "error_details": errors[:10]  # First 10 errors
        }
        
    except Exception as e:
        logger.error(f"Error uploading poles CSV: {e}")
        raise HTTPException(status_code=500, detail="Error processing CSV file")

# Statistics Endpoint

@app.get("/statistics")
async def get_statistics(db: PoleDataAccess = Depends(get_database)):
    """Get system statistics."""
    try:
        poles = db.get_all_poles()
        all_soil_samples = db.get_soil_samples()
        all_inspections = db.get_inspections()
        open_work_orders = db.get_work_orders(status='open')
        
        # Health status summary
        health_assessments = []
        priorities = {}
        
        for pole in poles:
            assessment = db.get_latest_health_assessment(pole.pole_id)
            if assessment:
                health_assessments.append(assessment)
                priority = assessment.maintenance_priority
                priorities[priority] = priorities.get(priority, 0) + 1
        
        avg_health = 0
        if health_assessments:
            avg_health = sum(a.overall_health_score for a in health_assessments) / len(health_assessments)
        
        return {
            "poles": {
                "total": len(poles),
                "by_type": {pole_type: len([p for p in poles if p.pole_type == pole_type]) 
                          for pole_type in set(p.pole_type for p in poles)}
            },
            "soil_samples": {
                "total": len(all_soil_samples)
            },
            "inspections": {
                "total": len(all_inspections)
            },
            "work_orders": {
                "open": len(open_work_orders)
            },
            "health": {
                "average_score": round(avg_health, 1),
                "assessments": len(health_assessments),
                "priorities": priorities
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating statistics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
