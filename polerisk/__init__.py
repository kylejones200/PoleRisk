"""
polerisk - Predictive utility pole failure analysis and maintenance
optimization platform

A comprehensive platform for assessing utility pole health, predicting failures,
and optimizing maintenance schedules. Includes soil moisture analysis, risk scoring,
failure mode modeling, and economic lifecycle accounting.
"""

__version__ = "1.1.0"
__author__ = "Kyle T. Jones"
__email__ = "kyletjones@gmail.com"

# Main package imports - expose key functionality at top level
from .pole_health.assessment import PoleHealthAssessment, SoilConditionAnalyzer
from .pole_health.risk_scoring import PoleRiskScorer, MaintenanceScheduler
from .pole_health.pole_data import PoleInfo, SoilSample, PoleHealthMetrics

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "PoleHealthAssessment",
    "SoilConditionAnalyzer",
    "PoleRiskScorer",
    "MaintenanceScheduler",
    "PoleInfo",
    "SoilSample",
    "PoleHealthMetrics",
]
