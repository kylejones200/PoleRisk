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
from .scenario_stress_testing import (
    ScenarioType,
    ScenarioParameters,
    StressTestResult,
    ScenarioStressTester,
)
from .regulatory_artifacts import (
    RegulatoryFramework,
    ModelCard,
    MethodologySummary,
    VersionedAssumption,
    RegulatoryMapping,
    AuditReport,
    RegulatoryArtifactGenerator,
)
from .crew_logistics import (
    CrewType,
    WorkComplexity,
    Depot,
    Crew,
    WorkTask,
    CrewAssignment,
    CrewLogisticsOptimizer,
)
from .economic_accounting import (
    MaintenanceAction,
    MaintenanceCosts,
    AvoidedCosts,
    DeferredMaintenanceRisk,
    LifecycleCostAnalysis,
    EconomicLifecycleAccountant,
)
from .data_lineage import (
    DataSourceType,
    DatasetFingerprint,
    AssumptionEntry,
    ChangeLogEntry,
    DataLineage,
    DataLineageTracker,
)
from .integration_hooks import (
    IntegrationSystem,
    WorkOrderExport,
    PoleAssetExport,
    IntegrationAdapter,
    MaximoAdapter,
    SAPPMAdapter,
    ESRIAdapter,
    OMSAdapter,
    IntegrationManager,
)
from .fleet_validation import (
    ValidationMetrics,
    BacktestResult,
    FleetValidationReport,
    FleetValidationAnalyzer,
)
from .reference_studies import (
    StudyType,
    StudyMetrics,
    ReferenceStudy,
    ReferenceStudyManager,
)
from .ts2net_integration import (
    NetworkMethod,
    NetworkFeatures,
    TimeSeriesNetworkAnalysis,
    TS2NetAnalyzer,
    batch_analyze_time_series,
)

try:
    from .anomaly_detection_integration import (
        AnomalyDetectionMethod,
        AnomalyDetectionResult,
        EnsembleAnomalyResult,
        AnomalyDetector,
        EnsembleAnomalyDetector,
    )

    ANOMALY_DETECTION_AVAILABLE = True
except ImportError:
    ANOMALY_DETECTION_AVAILABLE = False

__all__ = [
    "PoleHealthAssessment",
    "SoilConditionAnalyzer",
    "PoleRiskScorer",
    "MaintenanceScheduler",
    "PoleInfo",
    "SoilSample",
    "FailureMode",
    "FailureModeRisk",
    "FailureModeAnalysis",
    "FailureModeModel",
    "InspectionMethod",
    "InspectorProfile",
    "DefectType",
    "Defect",
    "InspectionResult",
    "InspectionRealismModel",
    "PoleClass",
    "EnvironmentType",
    "SurvivalCurve",
    "RemainingUsefulLife",
    "SurvivalAnalysisModel",
    "ScenarioType",
    "ScenarioParameters",
    "StressTestResult",
    "ScenarioStressTester",
    "RegulatoryFramework",
    "ModelCard",
    "MethodologySummary",
    "VersionedAssumption",
    "RegulatoryMapping",
    "AuditReport",
    "RegulatoryArtifactGenerator",
    "CrewType",
    "WorkComplexity",
    "Depot",
    "Crew",
    "WorkTask",
    "CrewAssignment",
    "CrewLogisticsOptimizer",
    "MaintenanceAction",
    "MaintenanceCosts",
    "AvoidedCosts",
    "DeferredMaintenanceRisk",
    "LifecycleCostAnalysis",
    "EconomicLifecycleAccountant",
    "DataSourceType",
    "DatasetFingerprint",
    "AssumptionEntry",
    "ChangeLogEntry",
    "DataLineage",
    "DataLineageTracker",
    "IntegrationSystem",
    "WorkOrderExport",
    "PoleAssetExport",
    "IntegrationAdapter",
    "MaximoAdapter",
    "SAPPMAdapter",
    "ESRIAdapter",
    "OMSAdapter",
    "IntegrationManager",
    "ValidationMetrics",
    "BacktestResult",
    "FleetValidationReport",
    "FleetValidationAnalyzer",
    "StudyType",
    "StudyMetrics",
    "ReferenceStudy",
    "ReferenceStudyManager",
    "NetworkMethod",
    "NetworkFeatures",
    "TimeSeriesNetworkAnalysis",
    "TS2NetAnalyzer",
    "batch_analyze_time_series",
]

if ANOMALY_DETECTION_AVAILABLE:
    __all__.extend(
        [
            "AnomalyDetectionMethod",
            "AnomalyDetectionResult",
            "EnsembleAnomalyResult",
            "AnomalyDetector",
            "EnsembleAnomalyDetector",
        ]
    )
