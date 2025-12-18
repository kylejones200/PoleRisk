# Repository Map: Utility Pole Soil Assessment System

## Project Summary

This Python package assesses soil conditions around utility poles to predict pole health, stability risks, and optimize maintenance scheduling. Originally designed for AMSR2 LPRM soil moisture data analysis, it has evolved into a comprehensive infrastructure health monitoring system combining soil science, geospatial analysis, machine learning, and web-based dashboards. The system provides multi-factor risk analysis, predictive maintenance scheduling, and real-time monitoring capabilities for utility pole fleets.

## Primary Goals

Enable utilities to make data-driven maintenance decisions by combining soil conditions, pole characteristics, and environmental factors into actionable intelligence. The system automates risk assessment, prioritizes maintenance tasks based on safety and reliability impact, and provides cost-benefit analysis of proactive versus reactive maintenance strategies. Core objectives include reducing pole failure rates, optimizing maintenance schedules, minimizing service interruptions, and extending infrastructure lifespan through predictive analytics.

## Architecture

Modular Python package with performance-critical components implemented in Rust, following a layered architecture from core data processing through machine learning to web interfaces. The system uses a plugin-based approach with separate modules for soil analysis (`core`), infrastructure assessment (`pole_health`), ML pipeline (`ml`), visualization (`visualization`), cloud integration (`cloud`), and web services (`web`). Data flows through standardized interfaces with SQLite for local storage, REST APIs for integration, and both CLI and web interfaces for user interaction.

## Folder Map

```
moisture/
├── Input/                       # Raw data inputs
│   ├── In-situ data/           # Field measurement data (.stm files)
│   ├── LPRM_NetCDF/            # Satellite soil moisture data (30 .nc files)
│   └── sample_*.csv            # Sample datasets for demonstration
├── Output/                      # Processed results and reports
│   ├── maintenance_report.txt  # Executive summary reports
│   ├── maintenance_schedule.csv # Prioritized maintenance tasks
│   └── pole_health_assessment.csv # Individual pole assessments
├── Analysis/                    # Generated visualizations and analysis
│   ├── pole_health_dashboard.png # Comprehensive overview dashboard
│   ├── soil_moisture_dashboard.html # Interactive web dashboard
│   └── [8 more analysis outputs] # Various plots and maps
├── soilmoisture/               # Main Python package
│   ├── core/                   # LPRM soil moisture processing
│   ├── pole_health/            # Utility pole assessment engine
│   ├── ml/                     # Machine learning models and CLI
│   ├── analysis/               # Statistical analysis functions
│   ├── visualization/          # Plotting and dashboard generation
│   ├── database/               # Data persistence and models
│   ├── web/                    # Web interface and API
│   ├── cloud/                  # AWS integration and batch processing
│   ├── weather/                # Weather data integration
│   ├── pipeline/               # Automated data ingestion
│   ├── utils/                  # Utility functions (geo, time)
│   └── tests/                  # Unit and integration tests
├── soilmoisture_rs/            # Rust performance extensions
│   ├── Cargo.toml              # Rust project configuration
│   └── src/                    # Rust source code
├── docs/                       # Sphinx documentation
│   └── source/                 # Documentation source files
└── [Config files]              # pyproject.toml, requirements.txt, etc.
```

## Key Modules

1. **main.py** - CLI entry point for pole health assessments and sample data generation
2. **dashboard_app.py** - Streamlit-based interactive web dashboard for monitoring and analysis
3. **api_server.py** - Flask/FastAPI REST API server for programmatic access
4. **soilmoisture/core/matching.py** - Core LPRM satellite data matching with in-situ measurements
5. **soilmoisture/core/lprm_utils.py** - AMSR2 LPRM soil moisture data processing utilities
6. **soilmoisture/pole_health/assessment.py** - Main pole health assessment engine and algorithms
7. **soilmoisture/pole_health/risk_scoring.py** - Risk scoring algorithms and maintenance scheduling
8. **soilmoisture/pole_health/pole_data.py** - Data models for poles, soil samples, and assessments
9. **soilmoisture/ml/models.py** - Machine learning models for prediction and anomaly detection
10. **soilmoisture/ml/cli.py** - Command-line interface for ML training and inference
11. **soilmoisture/ml/features.py** - Feature engineering for temporal and weather data
12. **soilmoisture/analysis/statistics.py** - Statistical functions with Rust acceleration
13. **soilmoisture/visualization/dashboard.py** - Dashboard generation and interactive plotting
14. **soilmoisture/visualization/maps.py** - Geographic visualization and interactive maps
15. **soilmoisture/database/models.py** - SQLite database models and ORM integration
16. **soilmoisture/web/api.py** - REST API endpoints for data upload and model management
17. **soilmoisture/cloud/aws_integration.py** - AWS S3 integration and cloud processing
18. **soilmoisture/pipeline/data_ingestion.py** - Automated data ingestion and processing workflows
19. **visualize_pole_health.py** - Batch visualization generator for comprehensive analysis
20. **launch_dashboard.py** - Dashboard launcher with data validation and setup
21. **launch_web_app.py** - Web application launcher with configuration options
22. **soilmoisture_rs/src/lib.rs** - Rust statistical functions for performance-critical operations

## Entry Points

### Command Line Interfaces
- **`python main.py`** - Main assessment CLI with options for poles, soil data, and output directories
- **`python main.py --create-sample-data`** - Generate demonstration datasets
- **`python -m soilmoisture.ml.cli train`** - ML model training with various algorithms
- **`python -m soilmoisture.ml.cli anomalies`** - Anomaly detection and outlier analysis
- **`python -m soilmoisture.ml.cli forecast`** - Time series forecasting and prediction
- **`python run_tests.py`** - Test runner with coverage and performance options

### Web Applications
- **`python launch_dashboard.py`** - Interactive Streamlit dashboard (port 8501)
- **`python dashboard_app.py`** - Direct Streamlit app launch
- **`python launch_web_app.py`** - Flask/FastAPI web application (port 5000)
- **`python api_server.py`** - REST API server for programmatic access

### Utilities
- **`python visualize_pole_health.py`** - Generate comprehensive static visualizations
- **`python demo_ml.py`** - ML capabilities demonstration with synthetic data
- **`python benchmark_rust_vs_python.py`** - Performance benchmarking between implementations

## Public APIs

### Core Soil Moisture Processing
```python
from soilmoisture.core import match_insitu_with_lprm, get_lprm_des, read_lprm_file
from soilmoisture.core import get_parameters, _find_nearest_valid
```

### Pole Health Assessment
```python
from soilmoisture.pole_health import PoleHealthAssessment, SoilConditionAnalyzer
from soilmoisture.pole_health import PoleRiskScorer, MaintenanceScheduler
from soilmoisture.pole_health import PoleInfo, SoilSample
```

### Machine Learning
```python
from soilmoisture.ml import SoilMoisturePredictor, AnomalyDetector, TimeSeriesForecaster
from soilmoisture.ml import FeatureEngineer, create_temporal_features, create_weather_features
```

### Analysis and Statistics
```python
from soilmoisture.analysis import calculate_rmse, calculate_correlation, calculate_mae
from soilmoisture.analysis import calculate_bias, calculate_ubrmse
```

### Data Pipeline
```python
from soilmoisture.pipeline import DataIngestionPipeline, ProcessingPipeline, BatchProcessor
from soilmoisture.pipeline import PipelineMonitor, DataQualityChecker
```

### Cloud Integration
```python
from soilmoisture.cloud import AWSProcessor, S3DataManager, CloudStorageManager
from soilmoisture.cloud import BatchProcessor, DistributedProcessor
```

### Visualization
```python
from soilmoisture.visualization import create_dashboard, plot_time_series, create_interactive_map
from soilmoisture.visualization import plot_correlation_matrix, generate_report_plots
```

## Data Flow

```
Raw Inputs → Core Processing → Analysis → ML/Assessment → Outputs
     ↓              ↓            ↓           ↓            ↓
1. CSV Files   2. LPRM Data   3. Stats   4. Models    5. Reports
   - Poles        Processing     - RMSE     - Health     - CSV
   - Soil         - NetCDF       - Corr     - Risk       - HTML
   - Weather      - Matching     - Bias     - Predict    - PNG
                  - Quality                 - Schedule   - JSON
```

### Detailed Flow
1. **Input Stage**: CSV files (poles, soil samples) and NetCDF satellite data loaded via pandas/netCDF4
2. **Core Processing**: LPRM satellite data matched with in-situ measurements using spatial-temporal algorithms
3. **Analysis Stage**: Statistical analysis (RMSE, correlation, bias) computed with Rust acceleration
4. **ML/Assessment Stage**: Machine learning models predict failures, assess health scores, calculate risk ratings
5. **Output Stage**: Generate maintenance schedules, health assessments, visualizations, and executive reports
6. **Storage**: Results cached in SQLite database for dashboard queries and historical analysis
7. **Visualization**: Interactive dashboards, static plots, and geographic maps generated on demand

## Config

### Primary Configuration
- **`pyproject.toml`** - Project metadata, dependencies, tool configurations (black, isort, pytest)
- **`requirements.txt`** - Production dependencies with version constraints
- **`requirements-dev.txt`** - Development dependencies (testing, linting, docs)
- **`setup.cfg`** - Legacy setuptools configuration and tool settings

### Runtime Configuration
- **`.env.weather`** - Weather API keys and service configuration
- **`docker-compose.yml`** - Multi-container deployment configuration
- **`Dockerfile`** - Container build instructions with Python and Rust
- **`nginx.conf`** - Web server configuration for production deployment

### Documentation
- **`docs/source/conf.py`** - Sphinx documentation configuration
- **`.readthedocs.yaml`** - ReadTheDocs build configuration

## Runtime

### Python Environment
- **Python Version**: 3.8+ (supports 3.8, 3.9, 3.10)
- **Package Manager**: pip with setuptools build backend

### Core Dependencies
- **Scientific Computing**: numpy>=1.21.0, pandas>=1.3.0, scipy>=1.7.0
- **Geospatial**: netCDF4>=1.5.7, cartopy>=0.19.0
- **Visualization**: matplotlib>=3.4.0, seaborn>=0.11.0, plotly>=5.0.0, folium>=0.12.0
- **Machine Learning**: scikit-learn>=1.0.0, tensorflow>=2.8.0 (optional)

### Optional Dependencies
- **Performance Extension**: `pip install -e ".[performance]"` - Numba JIT (2-50x speedup, **recommended**)
- **ML Extension**: `pip install -e ".[ml]"` - TensorFlow, PyTorch, XGBoost
- **Web Extension**: `pip install -e ".[web]"` - Flask, FastAPI, Gunicorn
- **Cloud Extension**: `pip install -e ".[cloud]"` - boto3, Docker, Dask
- **Dev Extension**: `pip install -e ".[dev]"` - pytest, black, mypy, sphinx
- **Complete Stack**: `pip install -e ".[all]"` - All optional features

### Performance Acceleration (Three-Tier Strategy)
- **Tier 1 - Rust Extensions**: 5-15x speedup on statistical functions (requires Rust toolchain)
- **Tier 2 - Numba JIT**: 2-50x speedup on geospatial operations (simple pip install)
- **Tier 3 - Pure Python**: Automatic fallback, guaranteed functionality

**Recommended**: Install `[performance]` tier for optimal speed without compilation complexity.

## Tests

### Test Execution
```bash
# Run all tests with coverage
python run_tests.py --type all --coverage

# Run specific test types
python run_tests.py --type unit
python run_tests.py --type integration  
python run_tests.py --type performance

# Direct pytest execution
pytest tests/ --cov=soilmoisture --cov-report=term-missing
```

### Test Structure
- **Unit Tests**: `tests/test_*.py` - Individual module testing with mocks and fixtures
- **Integration Tests**: Cross-module functionality and end-to-end workflows
- **Performance Tests**: Benchmarking with pytest-benchmark, marked with `@pytest.mark.performance`
- **Rust Integration**: `tests/test_rust_integration_rs.py` - Python-Rust interface validation

### Test Fixtures
- **`tests/data/`** - Sample .stm files and test datasets
- **Synthetic Data**: Property-based testing with Hypothesis for edge cases
- **Mock Objects**: Fake NetCDF datasets and API responses for isolated testing
- **Temporary Directories**: `tmp_path` fixtures for file I/O testing

### Coverage Requirements
- Source coverage target: `soilmoisture/` package
- Exclusions: `**/tests/*`, `**/__init__.py`, debug and error handling code
- CI Integration: Coverage reports integrated with development workflow

## Examples

### Basic Pole Assessment
```bash
# Generate sample data and run assessment
python main.py --create-sample-data
python main.py --poles Input/sample_poles.csv --soil Input/sample_soil_data.csv
python visualize_pole_health.py
```

### Interactive Dashboard
```bash
# Launch web dashboard for monitoring
python launch_dashboard.py
# Opens browser to http://localhost:8501 with interactive visualizations
```

### Machine Learning Workflow
```bash
# Train predictive models
python -m soilmoisture.ml.cli train --data-file Input/sample_soil_data.csv --model-type random_forest

# Detect anomalies in soil conditions  
python -m soilmoisture.ml.cli anomalies --data-file Input/sample_soil_data.csv --method isolation_forest

# Generate 7-day forecasts
python -m soilmoisture.ml.cli forecast --data-file Input/sample_soil_data.csv --forecast-days 7
```

### API Integration
```python
import requests

# Upload data via REST API
files = {'file': open('pole_data.csv', 'rb')}
response = requests.post('http://localhost:5000/api/upload', files=files)

# Train model programmatically
model_config = {'filename': 'pole_data.csv', 'model_type': 'random_forest'}
model = requests.post('http://localhost:5000/api/train-model', json=model_config)
```

## Known Issues

### Technical Debt
- **Dual Purpose Architecture**: System serves both soil moisture research and utility infrastructure - may benefit from clearer separation of concerns
- **Rust Integration Complexity**: Performance extensions require Rust toolchain, complicating deployment and development setup
- **Configuration Fragmentation**: Multiple config files (.env, pyproject.toml, setup.cfg) could be consolidated

### Scalability Concerns
- **SQLite Limitations**: Local database may not scale for large utility fleets - consider PostgreSQL migration
- **Memory Usage**: Large NetCDF files and ML models may exceed memory limits for extensive datasets
- **Processing Performance**: Batch processing could benefit from distributed computing for real-time monitoring

### Deployment Challenges
- **Docker Complexity**: Multi-language container builds increase deployment complexity and build times
- **Dependency Conflicts**: TensorFlow and other heavy ML dependencies may conflict in some environments
- **Documentation Gaps**: API documentation could be more comprehensive for integration developers

### Development Workflow
- **Test Coverage**: Some modules lack comprehensive test coverage, particularly cloud and web components
- **Performance Monitoring**: Limited profiling and performance regression testing in CI pipeline
- **Version Management**: Rust and Python version synchronization needs clearer workflow

### Data Quality
- **Input Validation**: Limited validation of input CSV formats may cause runtime errors
- **Missing Data Handling**: Inconsistent approaches to handling missing soil measurements across modules
- **Time Zone Handling**: UTC/local time conversions may introduce errors in global deployments

---
*Generated on 2025-12-18 | Repository contains 92 Python files across 3 directory levels*
