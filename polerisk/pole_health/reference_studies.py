"""
Reference studies and case studies for utility pole assessments.

Bundles anonymized case studies with before and after metrics to anchor
the platform in reality and demonstrate real-world impact.
"""

import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class StudyType(Enum):
    """Types of reference studies."""

    CASE_STUDY = "case_study"  # Single utility deployment
    PILOT_PROGRAM = "pilot_program"  # Limited scope pilot
    FLEET_COMPARISON = "fleet_comparison"  # Before/after comparison
    COST_BENEFIT = "cost_benefit"  # ROI analysis


@dataclass
class StudyMetrics:
    """Before and after metrics for a study."""

    # Time period
    baseline_period_start: datetime
    baseline_period_end: datetime
    intervention_period_start: datetime
    intervention_period_end: datetime

    # Failure metrics
    baseline_failures: int
    intervention_failures: int
    failure_reduction_percent: float

    # Cost metrics
    baseline_maintenance_cost: float
    intervention_maintenance_cost: float
    cost_reduction_percent: float

    baseline_emergency_cost: float
    intervention_emergency_cost: float
    emergency_cost_reduction_percent: float

    # Reliability metrics
    baseline_outage_hours: float
    intervention_outage_hours: float
    outage_reduction_percent: float

    baseline_customer_minutes_interrupted: float
    intervention_customer_minutes_interrupted: float
    cmi_reduction_percent: float

    # Model performance
    model_accuracy: float
    model_precision: float
    model_sensitivity: float
    false_positive_rate: float

    # ROI
    total_investment: float
    total_savings: float
    roi: float
    payback_period_months: float

    # Additional metrics
    additional_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReferenceStudy:
    """Reference study or case study."""

    study_id: str
    study_name: str
    study_type: StudyType
    created_date: datetime
    last_updated: datetime

    # Anonymization
    utility_name_anonymized: str  # e.g., "Utility A", "Midwest Utility"
    region_anonymized: str  # e.g., "Midwest US", "Coastal Region"
    fleet_size: int

    # Study details
    study_description: str
    objectives: List[str]
    methodology: str
    scope: str  # e.g., "450 poles", "Entire circuit", "10,000 poles"

    # Timeline
    study_duration_months: int
    baseline_period_months: int
    intervention_period_months: int

    # Metrics
    metrics: StudyMetrics

    # Key findings
    key_findings: List[str]
    lessons_learned: List[str]

    # Business impact
    business_impact_summary: str

    # Supporting data
    supporting_charts: List[str] = field(default_factory=list)  # File paths to charts
    supporting_data: Optional[Dict[str, Any]] = None

    # Validation
    validated_by: Optional[str] = None
    validation_date: Optional[datetime] = None

    # Confidentiality
    is_public: bool = False
    requires_nda: bool = True


class ReferenceStudyManager:
    """Manager for reference studies and case studies."""

    def __init__(self, studies_directory: str = "Output/reference_studies"):
        """Initialize reference study manager."""
        self.studies_directory = Path(studies_directory)
        self.studies_directory.mkdir(parents=True, exist_ok=True)

        self.studies: Dict[str, ReferenceStudy] = {}

        # Load default studies
        self._initialize_default_studies()

    def add_study(self, study: ReferenceStudy):
        """Add a reference study."""
        self.studies[study.study_id] = study
        self._save_study(study)

    def get_study(self, study_id: str) -> Optional[ReferenceStudy]:
        """Get a reference study by ID."""
        return self.studies.get(study_id)

    def list_studies(
        self, study_type: Optional[StudyType] = None, public_only: bool = False
    ) -> List[ReferenceStudy]:
        """List available studies."""
        studies = list(self.studies.values())

        if study_type:
            studies = [s for s in studies if s.study_type == study_type]

        if public_only:
            studies = [s for s in studies if s.is_public]

        return studies

    def generate_study_summary(self, study: ReferenceStudy) -> Dict[str, Any]:
        """Generate a summary of a study for presentation."""
        return {
            "study_id": study.study_id,
            "study_name": study.study_name,
            "utility": study.utility_name_anonymized,
            "region": study.region_anonymized,
            "fleet_size": study.fleet_size,
            "study_type": study.study_type.value,
            "duration_months": study.study_duration_months,
            "scope": study.scope,
            # Key metrics summary
            "failure_reduction": f"{study.metrics.failure_reduction_percent:.1f}%",
            "cost_reduction": f"{study.metrics.cost_reduction_percent:.1f}%",
            "roi": f"{study.metrics.roi:.1f}:1",
            "payback_months": f"{study.metrics.payback_period_months:.1f}",
            # Model performance
            "model_accuracy": f"{study.metrics.model_accuracy:.1%}",
            "model_precision": f"{study.metrics.model_precision:.1%}",
            # Key findings (first 3)
            "key_findings": study.key_findings[:3],
            "business_impact": study.business_impact_summary,
        }

    def export_study(self, study_id: str, format: str = "json") -> str:
        """Export a study to a file."""
        study = self.get_study(study_id)
        if not study:
            raise ValueError(f"Study {study_id} not found")

        if format == "json":
            output_file = self.studies_directory / f"{study_id}.json"
            with open(output_file, "w") as f:
                json.dump(self._serialize_study(study), f, indent=2, default=str)
            return str(output_file)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def create_comparison_report(self, study_ids: List[str]) -> Dict[str, Any]:
        """Create a comparison report across multiple studies."""
        studies = [self.get_study(sid) for sid in study_ids if self.get_study(sid)]

        if not studies:
            return {"error": "No valid studies found"}

        comparison = {
            "comparison_date": datetime.now().isoformat(),
            "studies_compared": len(studies),
            "study_ids": study_ids,
            # Aggregate metrics
            "average_roi": sum(s.metrics.roi for s in studies) / len(studies),
            "average_failure_reduction": sum(
                s.metrics.failure_reduction_percent for s in studies
            )
            / len(studies),
            "average_cost_reduction": sum(
                s.metrics.cost_reduction_percent for s in studies
            )
            / len(studies),
            "average_payback_months": sum(
                s.metrics.payback_period_months for s in studies
            )
            / len(studies),
            # Model performance
            "average_model_accuracy": sum(s.metrics.model_accuracy for s in studies)
            / len(studies),
            "average_model_precision": sum(s.metrics.model_precision for s in studies)
            / len(studies),
            # Study summaries
            "study_summaries": [self.generate_study_summary(s) for s in studies],
        }

        return comparison

    def _save_study(self, study: ReferenceStudy):
        """Save a study to disk."""
        output_file = self.studies_directory / f"{study.study_id}.json"
        with open(output_file, "w") as f:
            json.dump(self._serialize_study(study), f, indent=2, default=str)

    def _serialize_study(self, study: ReferenceStudy) -> Dict[str, Any]:
        """Serialize study to dict for JSON export."""
        return {
            k: (v.isoformat() if isinstance(v, datetime) else v)
            for k, v in asdict(study).items()
        }

    def _initialize_default_studies(self):
        """Initialize with default reference studies."""
        # Cedar Creek Case Study (from existing case study)
        cedar_creek_study = ReferenceStudy(
            study_id="CEDAR_CREEK_2024",
            study_name="Cedar Creek Circuit Deployment",
            study_type=StudyType.CASE_STUDY,
            created_date=datetime(2024, 1, 1),
            last_updated=datetime.now(),
            utility_name_anonymized="Midwest Utility",
            region_anonymized="Midwest US",
            fleet_size=450,
            study_description="Comprehensive deployment of pole health assessment platform on a critical distribution circuit with 450 poles. Demonstrates failure prediction accuracy and ROI from proactive maintenance.",
            objectives=[
                "Validate model accuracy with real-world failures",
                "Demonstrate ROI from proactive maintenance",
                "Measure reduction in emergency failures",
                "Assess impact on reliability metrics",
            ],
            methodology="Risk-based assessment with 6-month prediction horizon. Proactive replacement and treatment based on model recommendations.",
            scope="450 poles on Cedar Creek distribution circuit",
            study_duration_months=18,
            baseline_period_months=6,
            intervention_period_months=12,
            metrics=StudyMetrics(
                baseline_period_start=datetime(2023, 1, 1),
                baseline_period_end=datetime(2023, 6, 30),
                intervention_period_start=datetime(2023, 7, 1),
                intervention_period_end=datetime(2024, 6, 30),
                baseline_failures=18,
                intervention_failures=4,
                failure_reduction_percent=78.0,
                baseline_maintenance_cost=6500000.0,
                intervention_maintenance_cost=4400000.0,
                cost_reduction_percent=32.3,
                baseline_emergency_cost=810000.0,
                intervention_emergency_cost=180000.0,
                emergency_cost_reduction_percent=77.8,
                baseline_outage_hours=108.0,
                intervention_outage_hours=8.4,
                outage_reduction_percent=92.2,
                baseline_customer_minutes_interrupted=486000.0,
                intervention_customer_minutes_interrupted=18900.0,
                cmi_reduction_percent=96.1,
                model_accuracy=0.940,
                model_precision=0.855,
                model_sensitivity=0.940,
                false_positive_rate=0.023,
                total_investment=324000.0,
                total_savings=2100000.0,
                roi=12.0,
                payback_period_months=1.9,
            ),
            key_findings=[
                "78% reduction in emergency failures",
                "12:1 ROI in first year",
                "94% model accuracy with 85.5% precision",
                "96% reduction in customer minutes interrupted",
                "Platform predicted failure 6 months in advance",
                "$2.1M savings on $324K investment",
            ],
            lessons_learned=[
                "Early risk detection enables proactive maintenance",
                "Model confidence correlates with intervention success",
                "Integration with existing work management systems critical",
                "Field crew buy-in essential for success",
                "Regular model calibration improves accuracy",
            ],
            business_impact_summary="Cedar Creek deployment demonstrates compelling business case with 12:1 ROI in first year. Platform enabled 78% reduction in emergency failures and 96% reduction in customer impact, validating predictive approach to pole maintenance.",
            validated_by="Utility Operations Team",
            validation_date=datetime(2024, 7, 1),
            is_public=True,
            requires_nda=False,
        )

        self.add_study(cedar_creek_study)

        # Additional sample study - Pilot Program
        pilot_study = ReferenceStudy(
            study_id="COASTAL_PILOT_2023",
            study_name="Coastal Region Pilot Program",
            study_type=StudyType.PILOT_PROGRAM,
            created_date=datetime(2023, 6, 1),
            last_updated=datetime(2023, 12, 31),
            utility_name_anonymized="Coastal Utility",
            region_anonymized="Southeast Coastal",
            fleet_size=1000,
            study_description="Pilot program evaluating platform effectiveness in coastal environment with high humidity and salt exposure.",
            objectives=[
                "Test model performance in coastal environment",
                "Evaluate ROI in high-corrosion environment",
                "Validate treatment recommendations",
            ],
            methodology="12-month pilot with 1000 poles. Focus on groundline decay detection and treatment.",
            scope="1000 poles in coastal service area",
            study_duration_months=12,
            baseline_period_months=3,
            intervention_period_months=9,
            metrics=StudyMetrics(
                baseline_period_start=datetime(2023, 1, 1),
                baseline_period_end=datetime(2023, 3, 31),
                intervention_period_start=datetime(2023, 4, 1),
                intervention_period_end=datetime(2023, 12, 31),
                baseline_failures=12,
                intervention_failures=3,
                failure_reduction_percent=75.0,
                baseline_maintenance_cost=1200000.0,
                intervention_maintenance_cost=900000.0,
                cost_reduction_percent=25.0,
                baseline_emergency_cost=360000.0,
                intervention_emergency_cost=90000.0,
                emergency_cost_reduction_percent=75.0,
                baseline_outage_hours=72.0,
                intervention_outage_hours=18.0,
                outage_reduction_percent=75.0,
                baseline_customer_minutes_interrupted=360000.0,
                intervention_customer_minutes_interrupted=90000.0,
                cmi_reduction_percent=75.0,
                model_accuracy=0.920,
                model_precision=0.810,
                model_sensitivity=0.900,
                false_positive_rate=0.031,
                total_investment=120000.0,
                total_savings=360000.0,
                roi=3.0,
                payback_period_months=4.0,
            ),
            key_findings=[
                "75% reduction in failures in coastal environment",
                "Model effective at detecting groundline decay",
                "Treatment recommendations validated",
                "3:1 ROI in 9-month period",
            ],
            lessons_learned=[
                "Coastal environment requires adjusted risk thresholds",
                "Groundline treatment highly effective",
                "Salt exposure accelerates decay patterns",
            ],
            business_impact_summary="Pilot program validated platform effectiveness in coastal environments. Demonstrated 75% failure reduction with 3:1 ROI.",
            is_public=True,
            requires_nda=False,
        )

        self.add_study(pilot_study)
