"""
Data lineage and trust framework for utility pole assessments.

Provides dataset fingerprints, assumption registries, and change logs.
Utilities demand traceability when models affect safety and rates.
"""

import hashlib
import json
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DataSourceType(Enum):
    """Types of data sources."""

    SATELLITE = "satellite"
    WEATHER_API = "weather_api"
    POLE_INVENTORY = "pole_inventory"
    INSPECTION_RECORDS = "inspection_records"
    MAINTENANCE_HISTORY = "maintenance_history"
    SOIL_SAMPLES = "soil_samples"
    SENSOR_DATA = "sensor_data"
    MANUAL_ENTRY = "manual_entry"


@dataclass
class DatasetFingerprint:
    """Fingerprint of a dataset for verification."""

    dataset_id: str
    dataset_name: str
    source_type: DataSourceType
    creation_date: datetime
    last_modified: datetime
    # Fingerprint data (required fields)
    record_count: int
    schema_hash: str  # Hash of column names and types
    content_hash: str  # Hash of actual data content
    statistical_fingerprint: Dict[str, float]  # Mean, std, min, max for numeric columns
    # Metadata (required fields)
    data_quality_score: float
    completeness: float  # Percentage of non-null values
    provenance: str  # Where the data came from
    # Versioning (required fields)
    version: str
    # Optional fields (with defaults) must come last
    parent_dataset_id: Optional[str] = None  # If derived from another dataset


@dataclass
class AssumptionEntry:
    """Entry in the assumption registry."""

    assumption_id: str
    assumption_name: str
    version: str
    effective_date: datetime
    # Details (required fields first)
    description: str
    rationale: str
    source: str  # "expert_judgment", "literature", "empirical", etc.
    confidence: str  # "high", "medium", "low"
    # Impact (required fields)
    used_in: List[str]  # List of models/calculations using this assumption
    # Optional fields (with defaults) must come last
    retired_date: Optional[datetime] = None
    sensitivity: Optional[float] = None  # How sensitive results are to this assumption
    supporting_documents: List[str] = field(default_factory=list)
    reviewed_by: Optional[str] = None
    review_date: Optional[datetime] = None


@dataclass
class ChangeLogEntry:
    """Entry in the change log."""

    change_id: str
    timestamp: datetime
    change_type: (
        str  # "data_update", "model_change", "assumption_change", "methodology_change"
    )
    # What changed (required fields)
    entity_type: str  # "dataset", "model", "assumption", "methodology"
    entity_id: str
    entity_name: str
    # Change details (required fields)
    description: str
    # Who made the change (required field)
    changed_by: str
    # Optional fields (with defaults) must come last
    previous_version: Optional[str] = None
    new_version: Optional[str] = None
    impacted_assessments: List[str] = field(default_factory=list)
    requires_revalidation: bool = False
    approved_by: Optional[str] = None
    change_hash: Optional[str] = None  # Hash of the change for integrity


@dataclass
class DataLineage:
    """Data lineage for a specific assessment result."""

    assessment_id: str
    assessment_date: datetime

    # Input datasets
    input_datasets: List[DatasetFingerprint]

    # Processing steps
    processing_steps: List[Dict[str, Any]]  # Each step with details

    # Models used
    models_used: List[Dict[str, str]]  # model_name, version, parameters

    # Assumptions used
    assumptions_used: List[str]  # Assumption IDs

    # Output
    output_hash: str  # Hash of the final assessment result

    # Full lineage hash
    lineage_hash: str


class DataLineageTracker:
    """Track data lineage and trust for assessments."""

    def __init__(self, registry_directory: str = "Output/lineage"):
        """Initialize data lineage tracker."""
        self.registry_directory = Path(registry_directory)
        self.registry_directory.mkdir(parents=True, exist_ok=True)

        # Registries
        self.dataset_registry: Dict[str, DatasetFingerprint] = {}
        self.assumption_registry: Dict[str, AssumptionEntry] = {}
        self.change_log: List[ChangeLogEntry] = []

        # Load existing registries if available
        self._load_registries()

    def create_dataset_fingerprint(
        self,
        dataset_id: str,
        dataset_name: str,
        source_type: DataSourceType,
        data: Any,  # DataFrame, dict, etc.
        metadata: Dict[str, Any] = None,
    ) -> DatasetFingerprint:
        """Create a fingerprint for a dataset."""

        metadata = metadata or {}

        # Calculate fingerprints
        if hasattr(data, "shape"):  # Pandas DataFrame
            record_count = len(data)
            schema_hash = self._hash_schema(data)
            content_hash = self._hash_content(data)
            statistical_fingerprint = self._calculate_statistical_fingerprint(data)
            completeness = (
                data.notna().sum().sum() / (len(data) * len(data.columns))
                if len(data) > 0
                else 0.0
            )
        elif isinstance(data, dict):
            record_count = len(data)
            schema_hash = self._hash_dict_schema(data)
            content_hash = self._hash_dict_content(data)
            statistical_fingerprint = {}
            completeness = 1.0  # Simplified
        else:
            # For other types, create minimal fingerprint
            record_count = 1
            schema_hash = hashlib.sha256(str(type(data)).encode()).hexdigest()
            content_hash = hashlib.sha256(str(data).encode()).hexdigest()
            statistical_fingerprint = {}
            completeness = 1.0

        fingerprint = DatasetFingerprint(
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            source_type=source_type,
            creation_date=metadata.get("creation_date", datetime.now()),
            last_modified=datetime.now(),
            record_count=record_count,
            schema_hash=schema_hash,
            content_hash=content_hash,
            statistical_fingerprint=statistical_fingerprint,
            data_quality_score=metadata.get("data_quality_score", 0.8),
            completeness=completeness,
            provenance=metadata.get("provenance", "unknown"),
            version=metadata.get("version", "1.0.0"),
            parent_dataset_id=metadata.get("parent_dataset_id"),
        )

        # Register
        self.dataset_registry[dataset_id] = fingerprint
        self._save_registry("datasets")

        # Log change
        self._log_change(
            change_type="data_update",
            entity_type="dataset",
            entity_id=dataset_id,
            entity_name=dataset_name,
            description=f"Dataset fingerprint created: {dataset_name}",
            changed_by=metadata.get("created_by", "system"),
        )

        return fingerprint

    def register_assumption(self, assumption: AssumptionEntry) -> str:
        """Register an assumption in the registry."""
        self.assumption_registry[assumption.assumption_id] = assumption
        self._save_registry("assumptions")

        # Log change
        self._log_change(
            change_type="assumption_change",
            entity_type="assumption",
            entity_id=assumption.assumption_id,
            entity_name=assumption.assumption_name,
            description=f"Assumption registered: {assumption.assumption_name}",
            changed_by=assumption.reviewed_by or "system",
        )

        return assumption.assumption_id

    def create_data_lineage(
        self,
        assessment_id: str,
        input_datasets: List[DatasetFingerprint],
        processing_steps: List[Dict[str, Any]],
        models_used: List[Dict[str, str]],
        assumptions_used: List[str],
        output_data: Any,
    ) -> DataLineage:
        """Create data lineage for an assessment."""

        # Hash the output
        output_hash = (
            self._hash_content(output_data)
            if hasattr(output_data, "shape")
            else hashlib.sha256(str(output_data).encode()).hexdigest()
        )

        # Create lineage
        lineage_data = {
            "assessment_id": assessment_id,
            "assessment_date": datetime.now().isoformat(),
            "input_datasets": [asdict(ds) for ds in input_datasets],
            "processing_steps": processing_steps,
            "models_used": models_used,
            "assumptions_used": assumptions_used,
            "output_hash": output_hash,
        }

        lineage_hash = self._hash_dict_content(lineage_data)

        lineage = DataLineage(
            assessment_id=assessment_id,
            assessment_date=datetime.now(),
            input_datasets=input_datasets,
            processing_steps=processing_steps,
            models_used=models_used,
            assumptions_used=assumptions_used,
            output_hash=output_hash,
            lineage_hash=lineage_hash,
        )

        # Save lineage
        lineage_file = self.registry_directory / f"lineage_{assessment_id}.json"
        with open(lineage_file, "w") as f:
            json.dump(
                self._serialize_datetimes(asdict(lineage)), f, indent=2, default=str
            )

        return lineage

    def verify_dataset_integrity(
        self, dataset_id: str, current_data: Any
    ) -> Dict[str, Any]:
        """Verify dataset integrity by comparing current data to fingerprint."""
        if dataset_id not in self.dataset_registry:
            return {"verified": False, "reason": "dataset_not_registered"}

        stored_fingerprint = self.dataset_registry[dataset_id]

        # Recalculate fingerprints
        if hasattr(current_data, "shape"):
            current_schema_hash = self._hash_schema(current_data)
            current_content_hash = self._hash_content(current_data)
        else:
            return {"verified": False, "reason": "unsupported_data_type"}

        schema_match = current_schema_hash == stored_fingerprint.schema_hash
        content_match = current_content_hash == stored_fingerprint.content_hash

        return {
            "verified": schema_match and content_match,
            "schema_match": schema_match,
            "content_match": content_match,
            "stored_fingerprint": asdict(stored_fingerprint),
            "verification_date": datetime.now().isoformat(),
        }

    def get_change_log(
        self,
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[ChangeLogEntry]:
        """Get filtered change log entries."""
        filtered = self.change_log

        if entity_type:
            filtered = [e for e in filtered if e.entity_type == entity_type]

        if entity_id:
            filtered = [e for e in filtered if e.entity_id == entity_id]

        if start_date:
            filtered = [e for e in filtered if e.timestamp >= start_date]

        if end_date:
            filtered = [e for e in filtered if e.timestamp <= end_date]

        return sorted(filtered, key=lambda x: x.timestamp, reverse=True)

    def get_assumption_dependencies(self, assessment_id: str) -> Dict[str, Any]:
        """Get all assumptions and their dependencies for an assessment."""
        # Load lineage if available
        lineage_file = self.registry_directory / f"lineage_{assessment_id}.json"
        if not lineage_file.exists():
            return {"error": "lineage_not_found"}

        with open(lineage_file, "r") as f:
            lineage_data = json.load(f)

        assumption_ids = lineage_data.get("assumptions_used", [])
        assumptions = {
            assump_id: asdict(self.assumption_registry[assump_id])
            for assump_id in assumption_ids
            if assump_id in self.assumption_registry
        }

        return {
            "assessment_id": assessment_id,
            "assumptions": assumptions,
            "assumption_count": len(assumptions),
        }

    def _hash_schema(self, df) -> str:
        """Hash the schema of a DataFrame."""
        schema_str = json.dumps(
            {
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            },
            sort_keys=True,
        )
        return hashlib.sha256(schema_str.encode()).hexdigest()

    def _hash_content(self, df) -> str:
        """Hash the content of a DataFrame."""
        # Convert to string representation and hash
        content_str = df.to_string()
        return hashlib.sha256(content_str.encode()).hexdigest()

    def _hash_dict_schema(self, data: dict) -> str:
        """Hash the schema of a dict."""
        schema_str = json.dumps(list(data.keys()), sort_keys=True)
        return hashlib.sha256(schema_str.encode()).hexdigest()

    def _hash_dict_content(self, data: dict) -> str:
        """Hash the content of a dict."""
        content_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(content_str.encode()).hexdigest()

    def _calculate_statistical_fingerprint(self, df) -> Dict[str, float]:
        """Calculate statistical fingerprint of numeric columns."""
        fingerprint = {}
        numeric_cols = df.select_dtypes(include=["number"]).columns

        for col in numeric_cols:
            fingerprint[col] = {
                "mean": float(df[col].mean()) if not df[col].isna().all() else 0.0,
                "std": float(df[col].std()) if not df[col].isna().all() else 0.0,
                "min": float(df[col].min()) if not df[col].isna().all() else 0.0,
                "max": float(df[col].max()) if not df[col].isna().all() else 0.0,
            }

        return fingerprint

    def _log_change(
        self,
        change_type: str,
        entity_type: str,
        entity_id: str,
        entity_name: str,
        description: str,
        changed_by: str,
        previous_version: Optional[str] = None,
        new_version: Optional[str] = None,
    ):
        """Log a change to the change log."""
        change_id = f"CHANGE_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{entity_id}"

        change_data = {
            "change_id": change_id,
            "timestamp": datetime.now().isoformat(),
            "change_type": change_type,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "entity_name": entity_name,
            "description": description,
            "previous_version": previous_version,
            "new_version": new_version,
            "changed_by": changed_by,
        }

        change_hash = hashlib.sha256(
            json.dumps(change_data, sort_keys=True).encode()
        ).hexdigest()
        change_data["change_hash"] = change_hash

        entry = ChangeLogEntry(
            change_id=change_id,
            timestamp=datetime.now(),
            change_type=change_type,
            entity_type=entity_type,
            entity_id=entity_id,
            entity_name=entity_name,
            description=description,
            previous_version=previous_version,
            new_version=new_version,
            changed_by=changed_by,
            change_hash=change_hash,
        )

        self.change_log.append(entry)
        self._save_registry("changelog")

    def _save_registry(self, registry_type: str):
        """Save a registry to disk."""
        if registry_type == "datasets":
            data = {k: asdict(v) for k, v in self.dataset_registry.items()}
            filename = "dataset_registry.json"
        elif registry_type == "assumptions":
            data = {k: asdict(v) for k, v in self.assumption_registry.items()}
            filename = "assumption_registry.json"
        elif registry_type == "changelog":
            data = [asdict(entry) for entry in self.change_log]
            filename = "changelog.json"
        else:
            return

        filepath = self.registry_directory / filename
        with open(filepath, "w") as f:
            json.dump(self._serialize_datetimes(data), f, indent=2, default=str)

    def _load_registries(self):
        """Load registries from disk if they exist."""
        # Load dataset registry
        dataset_file = self.registry_directory / "dataset_registry.json"
        if dataset_file.exists():
            try:
                with open(dataset_file, "r") as f:
                    data = json.load(f)
                    # Would need to reconstruct DatasetFingerprint objects
                    # Simplified for now
            except Exception as e:
                logger.warning(f"Could not load dataset registry: {e}")

        # Similar for other registries...

    def _serialize_datetimes(self, obj: Any) -> Any:
        """Recursively serialize datetime objects."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: self._serialize_datetimes(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_datetimes(item) for item in obj]
        else:
            return obj
