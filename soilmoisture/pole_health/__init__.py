"""
Utility Pole Health Assessment Module.

This module contains functions and classes for assessing utility pole health
based on soil conditions, pole characteristics, and environmental factors.
"""

from .assessment import PoleHealthAssessment, SoilConditionAnalyzer
from .risk_scoring import PoleRiskScorer, MaintenanceScheduler
from .pole_data import PoleInfo, SoilSample
from .failure_modes import (
    FailureMode,
    FailureModeRisk,
    FailureModeAnalysis,
    FailureModeModel,
)
from .inspection_realism import (
    InspectionMethod,
    InspectorProfile,
    DefectType,
    Defect,
    InspectionResult,
    InspectionRealismModel,
)
from .survival_analysis import (
    PoleClass,
    EnvironmentType,
    SurvivalCurve,
    RemainingUsefulLife,
    SurvivalAnalysisModel,
)

__all__ = [
    'PoleHealthAssessment',
    'SoilConditionAnalyzer', 
    'PoleRiskScorer',
    'MaintenanceScheduler',
    'PoleInfo',
    'SoilSample',
    'FailureMode',
    'FailureModeRisk',
    'FailureModeAnalysis',
    'FailureModeModel',
    'InspectionMethod',
    'InspectorProfile',
    'DefectType',
    'Defect',
    'InspectionResult',
    'InspectionRealismModel',
    'PoleClass',
    'EnvironmentType',
    'SurvivalCurve',
    'RemainingUsefulLife',
    'SurvivalAnalysisModel',
]
