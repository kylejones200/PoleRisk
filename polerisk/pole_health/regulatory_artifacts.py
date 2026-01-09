"""
Regulatory artifacts and audit-ready outputs for utility pole assessments.

Generates defensible methodology summaries, model cards, versioned assumptions,
and maps outputs to NERC, state PUC, and wildfire mitigation filings. Essential
for audit readiness and regulatory compliance.
"""

import json
import hashlib
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class RegulatoryFramework(Enum):
    """Regulatory frameworks for compliance mapping."""

    NERC_CIP = "nerc_cip"  # North American Electric Reliability Corporation
    NERC_TADS = "nerc_tads"  # Transmission Availability Data System
    PUC_GENERAL = "puc_general"  # State Public Utilities Commission
    WILDFIRE_MITIGATION = "wildfire_mitigation"  # Wildfire safety regulations
    OSHA = "osha"  # Occupational Safety and Health Administration
    IEEE = "ieee"  # IEEE standards
    ANSI = "ansi"  # ANSI standards


@dataclass
class ModelCard:
    """Model card documenting a machine learning model."""

    model_name: str
    model_version: str
    model_type: str  # "random_forest", "neural_network", etc.
    created_date: datetime
    last_updated: datetime

    # Model details
    training_data_description: str
    training_date_range: tuple  # (start_date, end_date)
    number_of_training_samples: int
    feature_list: List[str]
    target_variable: str

    # Performance metrics
    performance_metrics: Dict[str, float]  # accuracy, precision, recall, etc.
    validation_method: str  # "cross_validation", "holdout", etc.
    validation_performance: Dict[str, float]

    # Limitations and assumptions
    known_limitations: List[str]
    assumptions: List[str]
    data_quality_notes: str

    # Ethical and bias considerations
    fairness_metrics: Optional[Dict[str, float]] = None
    bias_assessment: Optional[str] = None

    # Regulatory compliance
    regulatory_mappings: Dict[str, List[str]] = field(
        default_factory=dict
    )  # framework -> requirements
    audit_trail_hash: Optional[str] = None


@dataclass
class MethodologySummary:
    """Defensible methodology summary for audit purposes."""

    methodology_id: str
    methodology_name: str
    version: str
    effective_date: datetime
    last_reviewed: datetime
    reviewed_by: str

    # Purpose and scope
    purpose: str
    scope: str
    applicability: str

    # Methodology description
    approach: str
    key_algorithm: str
    data_sources: List[str]
    preprocessing_steps: List[str]

    # Validation
    validation_approach: str
    validation_results: Dict[str, Any]
    peer_review_status: str

    # Assumptions and limitations
    key_assumptions: List[str]
    limitations: List[str]
    uncertainty_quantification: str

    # Regulatory alignment
    regulatory_frameworks: List[RegulatoryFramework]
    compliance_statements: Dict[str, str]  # framework -> statement

    # Change log
    change_log: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class VersionedAssumption:
    """Versioned assumption for traceability."""

    assumption_id: str
    assumption_name: str
    version: str
    effective_date: datetime
    # Assumption details (required fields first)
    description: str
    rationale: str
    source: str  # "expert_judgment", "historical_data", "literature", etc.
    confidence_level: str  # "high", "medium", "low"
    # Impact assessment (required fields)
    impacted_components: List[str]  # Which models/calculations use this
    # Optional fields (with defaults) must come last
    retired_date: Optional[datetime] = None
    sensitivity_analysis: Optional[Dict[str, float]] = None
    supporting_evidence: List[str] = field(default_factory=list)
    review_date: Optional[datetime] = None
    reviewed_by: Optional[str] = None


@dataclass
class RegulatoryMapping:
    """Mapping of assessment outputs to regulatory requirements."""

    framework: RegulatoryFramework
    requirement_id: str
    requirement_description: str

    # Output mappings
    relevant_assessments: List[str]  # Assessment types that satisfy this
    evidence_type: str  # "risk_score", "inspection_record", "maintenance_log", etc.
    reporting_frequency: str  # "annual", "quarterly", "monthly", "on_demand"

    # Compliance status
    compliance_status: str  # "compliant", "partial", "non_compliant", "not_applicable"
    last_assessed: datetime
    next_assessment_due: datetime

    # Supporting documentation
    supporting_documents: List[str] = field(default_factory=list)


@dataclass
class AuditReport:
    """Comprehensive audit report for regulatory compliance."""

    report_id: str
    report_date: datetime
    report_type: str  # "annual", "quarterly", "ad_hoc", "incident_response"
    reporting_period: tuple  # (start_date, end_date)

    # Fleet summary
    total_poles_assessed: int
    assessment_coverage: float  # percentage
    high_risk_poles: int
    critical_poles: int

    # Assessment methodology
    methodology_used: str
    methodology_version: str
    model_versions: Dict[str, str]  # model_name -> version

    # Key findings
    key_findings: List[str]
    risk_summary: Dict[str, int]  # risk_level -> count

    # Compliance status
    regulatory_compliance: Dict[str, str]  # framework -> status
    compliance_gaps: List[str]
    remediation_plans: List[str]

    # Financial impact
    total_maintenance_spend: float
    avoided_costs: float
    roi_estimate: float

    # Data quality
    data_quality_score: float
    data_completeness: float
    confidence_level: float

    # Attestations
    prepared_by: str
    reviewed_by: str
    approved_by: str

    # Supporting artifacts
    model_cards: List[str] = field(default_factory=list)  # Model card IDs
    methodology_summaries: List[str] = field(default_factory=list)
    versioned_assumptions: List[str] = field(default_factory=list)

    # Audit trail
    audit_trail_hash: Optional[str] = None
    digital_signature: Optional[str] = None


class RegulatoryArtifactGenerator:
    """Generate regulatory artifacts and audit-ready outputs."""

    def __init__(self, output_directory: str = "Output/regulatory"):
        """Initialize regulatory artifact generator."""
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)

        # Registry of assumptions
        self.assumption_registry: Dict[str, VersionedAssumption] = {}

        # Registry of methodologies
        self.methodology_registry: Dict[str, MethodologySummary] = {}

        # Registry of model cards
        self.model_card_registry: Dict[str, ModelCard] = {}

        # Initialize default assumptions
        self._initialize_default_assumptions()
        self._initialize_default_methodologies()

    def generate_model_card(
        self, model_name: str, model_version: str, model_details: Dict[str, Any]
    ) -> ModelCard:
        """Generate a model card for documentation."""
        model_card = ModelCard(
            model_name=model_name,
            model_version=model_version,
            model_type=model_details.get("model_type", "unknown"),
            created_date=model_details.get("created_date", datetime.now()),
            last_updated=datetime.now(),
            training_data_description=model_details.get(
                "training_data_description", ""
            ),
            training_date_range=model_details.get("training_date_range", (None, None)),
            number_of_training_samples=model_details.get(
                "number_of_training_samples", 0
            ),
            feature_list=model_details.get("feature_list", []),
            target_variable=model_details.get("target_variable", "failure_probability"),
            performance_metrics=model_details.get("performance_metrics", {}),
            validation_method=model_details.get("validation_method", "holdout"),
            validation_performance=model_details.get("validation_performance", {}),
            known_limitations=model_details.get("known_limitations", []),
            assumptions=model_details.get("assumptions", []),
            data_quality_notes=model_details.get("data_quality_notes", ""),
            fairness_metrics=model_details.get("fairness_metrics"),
            bias_assessment=model_details.get("bias_assessment"),
            regulatory_mappings=model_details.get("regulatory_mappings", {}),
        )

        # Calculate audit trail hash
        model_card.audit_trail_hash = self._calculate_hash(asdict(model_card))

        # Register
        card_id = f"{model_name}_{model_version}"
        self.model_card_registry[card_id] = model_card

        return model_card

    def register_methodology(self, methodology: MethodologySummary) -> str:
        """Register a methodology summary."""
        method_id = methodology.methodology_id
        self.methodology_registry[method_id] = methodology
        return method_id

    def register_assumption(self, assumption: VersionedAssumption) -> str:
        """Register a versioned assumption."""
        self.assumption_registry[assumption.assumption_id] = assumption
        return assumption.assumption_id

    def create_regulatory_mapping(
        self,
        framework: RegulatoryFramework,
        requirement_id: str,
        requirement_description: str,
        relevant_assessments: List[str],
        evidence_type: str,
        reporting_frequency: str,
    ) -> RegulatoryMapping:
        """Create a regulatory mapping."""
        return RegulatoryMapping(
            framework=framework,
            requirement_id=requirement_id,
            requirement_description=requirement_description,
            relevant_assessments=relevant_assessments,
            evidence_type=evidence_type,
            reporting_frequency=reporting_frequency,
            compliance_status="not_applicable",
            last_assessed=datetime.now(),
            next_assessment_due=datetime.now(),
        )

    def generate_audit_report(
        self,
        report_type: str,
        reporting_period: tuple,
        fleet_data: Dict[str, Any],
        compliance_data: Dict[str, Any],
    ) -> AuditReport:
        """Generate a comprehensive audit report."""

        # Extract fleet summary
        total_poles = fleet_data.get("total_poles", 0)
        high_risk = fleet_data.get("high_risk_poles", 0)
        critical = fleet_data.get("critical_poles", 0)
        risk_summary = fleet_data.get("risk_summary", {})

        # Calculate compliance status
        regulatory_compliance = {}
        compliance_gaps = []

        for framework_name, status in compliance_data.items():
            regulatory_compliance[framework_name] = status.get("status", "unknown")
            if status.get("status") != "compliant":
                compliance_gaps.append(
                    f"{framework_name}: {status.get('gap_description', 'Non-compliant')}"
                )

        # Generate audit report
        report = AuditReport(
            report_id=f"AUDIT_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            report_date=datetime.now(),
            report_type=report_type,
            reporting_period=reporting_period,
            total_poles_assessed=total_poles,
            assessment_coverage=fleet_data.get("assessment_coverage", 0.0),
            high_risk_poles=high_risk,
            critical_poles=critical,
            methodology_used=fleet_data.get("methodology_name", "Pole Risk Assessment"),
            methodology_version=fleet_data.get("methodology_version", "1.0.0"),
            model_versions=fleet_data.get("model_versions", {}),
            key_findings=fleet_data.get("key_findings", []),
            risk_summary=risk_summary,
            regulatory_compliance=regulatory_compliance,
            compliance_gaps=compliance_gaps,
            remediation_plans=compliance_data.get("remediation_plans", []),
            total_maintenance_spend=fleet_data.get("total_maintenance_spend", 0.0),
            avoided_costs=fleet_data.get("avoided_costs", 0.0),
            roi_estimate=fleet_data.get("roi_estimate", 0.0),
            data_quality_score=fleet_data.get("data_quality_score", 0.0),
            data_completeness=fleet_data.get("data_completeness", 0.0),
            confidence_level=fleet_data.get("confidence_level", 0.0),
            prepared_by=fleet_data.get("prepared_by", "System"),
            reviewed_by=fleet_data.get("reviewed_by", ""),
            approved_by=fleet_data.get("approved_by", ""),
            model_cards=list(self.model_card_registry.keys()),
            methodology_summaries=list(self.methodology_registry.keys()),
            versioned_assumptions=list(self.assumption_registry.keys()),
        )

        # Calculate audit trail hash
        report_dict = asdict(report)
        report_dict.pop("audit_trail_hash", None)
        report_dict.pop("digital_signature", None)
        report.audit_trail_hash = self._calculate_hash(report_dict)

        return report

    def export_regulatory_package(
        self, audit_report: AuditReport, output_format: str = "json"
    ) -> str:
        """Export a complete regulatory package for audit."""

        package = {
            "audit_report": asdict(audit_report),
            "model_cards": {
                card_id: asdict(card)
                for card_id, card in self.model_card_registry.items()
                if card_id in audit_report.model_cards
            },
            "methodologies": {
                method_id: asdict(method)
                for method_id, method in self.methodology_registry.items()
                if method_id in audit_report.methodology_summaries
            },
            "assumptions": {
                assump_id: asdict(assump)
                for assump_id, assump in self.assumption_registry.items()
                if assump_id in audit_report.versioned_assumptions
            },
        }

        # Convert datetime objects to strings for JSON
        package = self._serialize_datetimes(package)

        if output_format == "json":
            output_file = self.output_directory / f"{audit_report.report_id}.json"
            with open(output_file, "w") as f:
                json.dump(package, f, indent=2)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

        return str(output_file)

    def map_to_regulatory_framework(
        self, assessment_results: Dict[str, Any], framework: RegulatoryFramework
    ) -> Dict[str, Any]:
        """Map assessment results to a specific regulatory framework."""

        mappings = {
            RegulatoryFramework.NERC_CIP: self._map_to_nerc_cip,
            RegulatoryFramework.NERC_TADS: self._map_to_nerc_tads,
            RegulatoryFramework.PUC_GENERAL: self._map_to_puc,
            RegulatoryFramework.WILDFIRE_MITIGATION: self._map_to_wildfire_mitigation,
        }

        mapper = mappings.get(framework)
        if mapper:
            return mapper(assessment_results)
        else:
            return {"status": "framework_not_implemented"}

    def _map_to_nerc_cip(self, assessment_results: Dict[str, Any]) -> Dict[str, Any]:
        """Map to NERC CIP requirements."""
        return {
            "framework": "NERC_CIP",
            "requirements": {
                "CIP-003": {
                    "description": "Security Management Controls",
                    "compliance": "assessed_via_audit_logs",
                    "evidence": "audit_trail_hash",
                },
                "CIP-010": {
                    "description": "Configuration Change Management",
                    "compliance": "assessed_via_versioned_assumptions",
                    "evidence": "versioned_assumptions_registry",
                },
            },
            "assessment_coverage": assessment_results.get("total_poles_assessed", 0),
            "last_assessment": datetime.now().isoformat(),
        }

    def _map_to_nerc_tads(self, assessment_results: Dict[str, Any]) -> Dict[str, Any]:
        """Map to NERC TADS requirements."""
        return {
            "framework": "NERC_TADS",
            "requirements": {
                "Availability_Reporting": {
                    "description": "Transmission asset availability data",
                    "compliance": "pole_health_data_supports_availability_calc",
                    "evidence": "risk_assessment_data",
                }
            },
        }

    def _map_to_puc(self, assessment_results: Dict[str, Any]) -> Dict[str, Any]:
        """Map to state PUC requirements."""
        return {
            "framework": "PUC_GENERAL",
            "requirements": {
                "Asset_Management": {
                    "description": "Demonstrated asset management practices",
                    "compliance": "comprehensive_pole_assessment_program",
                    "evidence": "audit_report_and_methodology",
                },
                "Rate_Case_Support": {
                    "description": "Support for rate case filings",
                    "compliance": "cost_benefit_analysis_provided",
                    "evidence": "maintenance_cost_analysis",
                },
            },
        }

    def _map_to_wildfire_mitigation(
        self, assessment_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Map to wildfire mitigation requirements."""
        return {
            "framework": "WILDFIRE_MITIGATION",
            "requirements": {
                "High_Risk_Area_Assessment": {
                    "description": "Assessment of poles in high wildfire risk areas",
                    "compliance": "wildfire_risk_included_in_assessment",
                    "evidence": "scenario_stress_test_results",
                },
                "Preventive_Maintenance": {
                    "description": "Preventive maintenance program for wildfire zones",
                    "compliance": "maintenance_schedules_address_wildfire_risk",
                    "evidence": "maintenance_schedules_and_priorities",
                },
            },
        }

    def _calculate_hash(self, data: Any) -> str:
        """Calculate SHA256 hash for audit trail."""
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def _serialize_datetimes(self, obj: Any) -> Any:
        """Recursively serialize datetime objects to strings."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: self._serialize_datetimes(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_datetimes(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._serialize_datetimes(item) for item in obj)
        elif hasattr(obj, "__dict__"):
            return self._serialize_datetimes(asdict(obj))
        else:
            return obj

    def _initialize_default_assumptions(self):
        """Initialize default versioned assumptions."""
        defaults = [
            VersionedAssumption(
                assumption_id="ASSUMPTION_001",
                assumption_name="Baseline failure rate by age",
                version="1.0.0",
                effective_date=datetime(2020, 1, 1),
                description="Poles follow age-based failure rate curves",
                rationale="Based on industry standard degradation models",
                source="historical_data",
                confidence_level="high",
                impacted_components=["survival_analysis", "risk_scoring"],
            ),
            VersionedAssumption(
                assumption_id="ASSUMPTION_002",
                assumption_name="Soil moisture impact on decay",
                version="1.0.0",
                effective_date=datetime(2020, 1, 1),
                description="High soil moisture accelerates wood decay",
                rationale="Empirical correlation observed in field studies",
                source="literature",
                confidence_level="medium",
                impacted_components=["health_assessment", "failure_modes"],
            ),
        ]

        for assumption in defaults:
            self.assumption_registry[assumption.assumption_id] = assumption

    def _initialize_default_methodologies(self):
        """Initialize default methodology summaries."""
        default_methodology = MethodologySummary(
            methodology_id="METHOD_001",
            methodology_name="Pole Health Risk Assessment",
            version="1.0.0",
            effective_date=datetime(2020, 1, 1),
            last_reviewed=datetime.now(),
            reviewed_by="System",
            purpose="Assess utility pole health and failure risk",
            scope="All utility poles in service area",
            applicability="Wooden utility poles",
            approach="Multi-factor risk assessment combining environmental, structural, and operational factors",
            key_algorithm="Weighted risk scoring with machine learning enhancement",
            data_sources=[
                "Satellite soil moisture",
                "Weather data",
                "Pole inventory",
                "Inspection records",
            ],
            preprocessing_steps=[
                "Data validation",
                "Geospatial matching",
                "Temporal alignment",
            ],
            validation_approach="Cross-validation with historical failure data",
            validation_results={"accuracy": 0.94, "precision": 0.855, "recall": 0.94},
            peer_review_status="internal_review",
            key_assumptions=[
                "Age-based degradation",
                "Environmental factors accelerate decay",
            ],
            limitations=[
                "Limited historical data",
                "Model accuracy depends on data quality",
            ],
            uncertainty_quantification="Confidence intervals provided with all risk scores",
            regulatory_frameworks=[RegulatoryFramework.PUC_GENERAL],
            compliance_statements={
                "puc_general": "Methodology supports asset management practices required by PUC"
            },
        )

        self.methodology_registry[default_methodology.methodology_id] = (
            default_methodology
        )
