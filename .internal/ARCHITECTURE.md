# Technical Architecture: Pole Health Assessment Platform

## System Overview

The platform follows a **three-layer architecture** with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    RISK SCORING LAYER                       │
│  • Pole health assessment        • Maintenance scheduling   │  
│  • Risk prioritization          • Cost-benefit analysis    │
│  • Business rules               • Decision support         │
└─────────────────────────────────────────────────────────────┘
                               ↑
┌─────────────────────────────────────────────────────────────┐
│                  FEATURE ENGINEERING LAYER                  │
│  • ML feature extraction        • Temporal analysis        │
│  • Weather integration         • Asset aging models       │  
│  • Statistical analysis        • Anomaly detection        │
└─────────────────────────────────────────────────────────────┘
                               ↑
┌─────────────────────────────────────────────────────────────┐
│                ENVIRONMENTAL SIGNALS LAYER                  │
│  • Soil moisture processing     • Weather data ingestion   │
│  • Satellite data extraction   • In-situ measurements     │
│  • Data quality validation     • Geospatial matching      │
└─────────────────────────────────────────────────────────────┘
```

## Layer Responsibilities

### **Layer 1: Environmental Signals** (`signals/`)
**Purpose:** Extract and validate raw environmental measurements

**Responsibilities:**
- Ingest satellite soil moisture data (AMSR2 LPRM)
- Process in-situ weather station data  
- Validate data quality and spatial matching
- Normalize measurements to standard formats
- Handle missing data and outliers

**Key Modules:**
- `signals.soil_moisture` - LPRM processing pipeline
- `signals.weather` - Weather API integration
- `signals.quality` - Data validation and QC
- `signals.geospatial` - Location matching and interpolation

**Outputs:** Standardized environmental time series per location

---

### **Layer 2: Feature Engineering** (`features/`)
**Purpose:** Convert raw signals into engineering-relevant risk indicators

**Responsibilities:**
- Transform environmental data into risk features
- Calculate temporal trends and seasonality
- Generate asset-specific risk indicators
- Apply domain knowledge (soil science → structural engineering)
- Create ML-ready feature matrices

**Key Modules:**
- `features.environmental` - Soil moisture → corrosion risk
- `features.temporal` - Trend analysis and forecasting
- `features.structural` - Age and material risk factors  
- `features.ml` - Feature selection and engineering

**Outputs:** Risk-relevant feature vectors per pole

---

### **Layer 3: Risk Scoring** (`assessment/`)
**Purpose:** Generate business decisions from engineered features

**Responsibilities:**
- Calculate pole health scores (0-100)
- Predict failure probabilities
- Prioritize maintenance actions
- Optimize resource allocation
- Generate executive reports

**Key Modules:**
- `assessment.health` - Pole health scoring algorithms
- `assessment.risk` - Multi-factor risk modeling
- `assessment.maintenance` - Scheduling optimization
- `assessment.business` - ROI and cost-benefit analysis

**Outputs:** Actionable maintenance decisions and risk reports

## Data Flow Architecture

```
Raw Data → Signals Layer → Features Layer → Assessment Layer → Business Actions

[Satellite]     [Soil Moisture]    [Corrosion Risk]     [Health Score]      [Work Order]
[Weather API] → [Weather Data]  →  [Weather Features] → [Failure Risk]  →  [Maintenance]  
[Asset DB]      [Asset Properties] [Aging Models]      [Priority Rank]     [Budget Plan]
```

## Interface Contracts

### **Signal → Feature Interface**
```python
@dataclass
class EnvironmentalSignal:
    pole_id: str
    timestamp: datetime
    soil_moisture: float
    temperature: float
    precipitation: float
    data_quality: QualityFlag
```

### **Feature → Assessment Interface**  
```python
@dataclass
class RiskFeatures:
    pole_id: str
    environmental_risk: float    # 0-1 scale
    structural_risk: float       # 0-1 scale  
    chemical_risk: float         # 0-1 scale
    confidence: float           # 0-1 scale
    feature_vector: np.ndarray  # ML inputs
```

### **Assessment → Business Interface**
```python
@dataclass  
class PoleAssessment:
    pole_id: str
    health_score: int           # 0-100 scale
    failure_probability: float  # 0-1, next 12 months
    maintenance_priority: Priority  # CRITICAL/HIGH/MEDIUM/LOW
    recommended_action: MaintenanceAction
    cost_estimate: float
    confidence_interval: tuple
```

## Enterprise Integration Points

### **Data Ingestion**
- **GIS Systems:** Pole locations and characteristics
- **CMMS:** Maintenance history and work orders  
- **Weather APIs:** Real-time environmental data
- **Satellite Data:** Automated LPRM processing

### **Decision Support**
- **Work Management:** Priority maintenance lists
- **Asset Management:** Risk-based replacement planning
- **Operations:** Storm preparation and response
- **Executive Dashboards:** Fleet health metrics

## Scalability Design

### **Horizontal Scaling**
- Microservice architecture per layer
- Event-driven communication between layers
- Cloud-native deployment (Kubernetes)
- Auto-scaling based on data volume

### **Data Architecture**
- **Operational Store:** PostgreSQL for real-time queries
- **Analytics Store:** Time-series database for environmental data  
- **ML Platform:** Feature store for model training/inference
- **Data Lake:** Raw data archival and batch processing

## Security & Compliance

### **Data Protection**
- Encryption at rest and in transit
- Role-based access controls  
- Audit logging for all decisions
- GDPR/CCPA compliance framework

### **Reliability**
- 99.9% uptime SLA
- Automated failover and recovery
- Data backup and disaster recovery
- Performance monitoring and alerting

---

## Migration from Research Prototype

### **Current State Issues**
- Monolithic `soilmoisture` package mixing all layers
- Direct database access throughout codebase
- Multiple entry points with overlapping functionality
- Research-focused APIs, not business-focused

### **Target State Benefits**  
- Clear separation enables independent testing and deployment
- Business layer can evolve without affecting signal processing
- Multiple ML models can be deployed without system changes
- Enterprise integrations isolated from research algorithms

### **Migration Strategy**
1. **Extract Signal Layer** - Move LPRM/weather processing to `signals/`
2. **Create Feature Layer** - Refactor ML code into `features/` 
3. **Isolate Assessment Layer** - Clean business logic in `assessment/`
4. **Add Enterprise APIs** - REST/GraphQL interfaces for each layer
5. **Deploy Incrementally** - Gradual rollout with backward compatibility
