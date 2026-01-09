"""
Integration hooks and adapters for external systems.

Provides adapters for Maximo, SAP PM, ESRI, and OMS exports to enable
bidirectional data flow and increase adoption.
"""

import json
import csv
from typing import Dict, List, Optional, Any, Iterator
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class IntegrationSystem(Enum):
    """Supported integration systems."""

    MAXIMO = "maximo"
    SAP_PM = "sap_pm"
    ESRI = "esri"
    OMS = "oms"  # Outage Management System
    CUSTOM = "custom"


@dataclass
class WorkOrderExport:
    """Work order for export to CMMS systems."""

    work_order_id: str
    pole_id: str
    work_type: str  # "inspection", "repair", "replacement"
    priority: str
    description: str
    location: Dict[str, float]  # latitude, longitude
    scheduled_date: datetime
    estimated_duration_hours: float
    required_skills: List[str]
    required_equipment: List[str]
    materials_needed: Dict[str, float]
    status: str = "new"


@dataclass
class PoleAssetExport:
    """Pole asset data for export to GIS systems."""

    pole_id: str
    latitude: float
    longitude: float
    installation_date: datetime
    material: str
    height_ft: float
    class_rating: str
    risk_score: float
    health_score: float
    last_inspection_date: Optional[datetime] = None
    next_inspection_due: Optional[datetime] = None
    maintenance_status: str = "operational"


class IntegrationAdapter:
    """Base class for integration adapters."""

    def export_work_orders(self, work_orders: List[WorkOrderExport]) -> str:
        """Export work orders. Returns file path or connection status."""
        raise NotImplementedError

    def import_work_orders(self, source: str) -> List[Dict[str, Any]]:
        """Import work orders from external system."""
        raise NotImplementedError

    def export_pole_data(self, poles: List[PoleAssetExport]) -> str:
        """Export pole asset data."""
        raise NotImplementedError

    def import_pole_data(self, source: str) -> List[Dict[str, Any]]:
        """Import pole asset data from external system."""
        raise NotImplementedError


class MaximoAdapter(IntegrationAdapter):
    """Adapter for IBM Maximo CMMS system."""

    def __init__(
        self, api_endpoint: Optional[str] = None, api_key: Optional[str] = None
    ):
        """Initialize Maximo adapter."""
        self.api_endpoint = api_endpoint or "https://maximo.example.com/maximo/api"
        self.api_key = api_key
        self.format = "json"  # Maximo supports JSON and XML

    def export_work_orders(self, work_orders: List[WorkOrderExport]) -> str:
        """Export work orders to Maximo format."""
        maximo_workorders = []

        for wo in work_orders:
            maximo_wo = {
                "wonum": wo.work_order_id,
                "description": wo.description,
                "status": "WAPPR" if wo.status == "new" else wo.status.upper(),
                "location": wo.location.get("address", ""),
                "latitude": wo.location.get("latitude"),
                "longitude": wo.location.get("longitude"),
                "schedstart": wo.scheduled_date.isoformat(),
                "estdur": wo.estimated_duration_hours,
                "worktype": wo.work_type.upper(),
                "priority": self._map_priority_to_maximo(wo.priority),
                "assetnum": wo.pole_id,
                "pluspreqskill": ",".join(wo.required_skills),
                "pluspequip": ",".join(wo.required_equipment),
                "pluspmaterials": json.dumps(wo.materials_needed),
            }
            maximo_workorders.append(maximo_wo)

        # Export to JSON file (in production, would POST to Maximo API)
        output_file = (
            f"Output/maximo_workorders_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        export_data = {
            "workorder": maximo_workorders,
            "metadata": {
                "export_date": datetime.now().isoformat(),
                "system": "polerisk",
                "format_version": "1.0",
            },
        }

        with open(output_file, "w") as f:
            json.dump(export_data, f, indent=2)

        logger.info(
            f"Exported {len(work_orders)} work orders to Maximo format: {output_file}"
        )
        return output_file

    def import_work_orders(self, source: str) -> List[Dict[str, Any]]:
        """Import work orders from Maximo."""
        if source.startswith("http"):
            # Would make API call in production
            logger.warning("Maximo API import not yet implemented, using file import")
            source = "maximo_workorders_import.json"

        with open(source, "r") as f:
            data = json.load(f)

        work_orders = []
        for wo_data in data.get("workorder", []):
            work_orders.append(
                {
                    "work_order_id": wo_data.get("wonum"),
                    "pole_id": wo_data.get("assetnum"),
                    "work_type": wo_data.get("worktype", "").lower(),
                    "status": wo_data.get("status", "").lower(),
                    "description": wo_data.get("description"),
                    "scheduled_date": wo_data.get("schedstart"),
                    "priority": wo_data.get("priority"),
                }
            )

        return work_orders

    def _map_priority_to_maximo(self, priority: str) -> str:
        """Map internal priority to Maximo priority codes."""
        mapping = {"critical": "1", "high": "2", "medium": "3", "low": "4"}
        return mapping.get(priority.lower(), "3")

    def export_pole_data(self, poles: List[PoleAssetExport]) -> str:
        """Export pole data to Maximo asset format."""
        maximo_assets = []

        for pole in poles:
            asset = {
                "assetnum": pole.pole_id,
                "description": f"Utility Pole {pole.pole_id}",
                "latitude": pole.latitude,
                "longitude": pole.longitude,
                "pluspinstall": (
                    pole.installation_date.isoformat()
                    if pole.installation_date
                    else None
                ),
                "pluspmaterial": pole.material,
                "pluspheight": pole.height_ft,
                "pluspclass": pole.class_rating,
                "pluspriskscore": pole.risk_score,
                "plusphealthscore": pole.health_score,
                "plusplastinsp": (
                    pole.last_inspection_date.isoformat()
                    if pole.last_inspection_date
                    else None
                ),
                "pluspnextinsp": (
                    pole.next_inspection_due.isoformat()
                    if pole.next_inspection_due
                    else None
                ),
                "pluspmaintstat": pole.maintenance_status,
            }
            maximo_assets.append(asset)

        output_file = (
            f"Output/maximo_assets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump({"asset": maximo_assets}, f, indent=2)

        return output_file


class SAPPMAdapter(IntegrationAdapter):
    """Adapter for SAP Plant Maintenance (PM) module."""

    def __init__(self, sap_system: Optional[str] = None):
        """Initialize SAP PM adapter."""
        self.sap_system = sap_system or "SAP_PROD"

    def export_work_orders(self, work_orders: List[WorkOrderExport]) -> str:
        """Export work orders to SAP PM format."""
        sap_orders = []

        for wo in work_orders:
            sap_order = {
                "ORDERID": wo.work_order_id,
                "EQUIPMENT": wo.pole_id,
                "ORDER_TYPE": self._map_work_type_to_sap(wo.work_type),
                "PRIORITY": wo.priority.upper(),
                "SHORT_TEXT": wo.description[:40],  # SAP limit
                "LONG_TEXT": wo.description,
                "PLANNED_START": wo.scheduled_date.strftime("%Y%m%d"),
                "PLANNED_DURATION": int(
                    wo.estimated_duration_hours * 60
                ),  # SAP uses minutes
                "LATITUDE": wo.location.get("latitude"),
                "LONGITUDE": wo.location.get("longitude"),
                "REQUIRED_SKILLS": ",".join(wo.required_skills),
                "MATERIALS": json.dumps(wo.materials_needed),
            }
            sap_orders.append(sap_order)

        # Export to CSV (SAP often uses CSV for bulk imports)
        output_file = (
            f"Output/sap_pm_orders_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        if sap_orders:
            with open(output_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=sap_orders[0].keys())
                writer.writeheader()
                writer.writerows(sap_orders)

        logger.info(
            f"Exported {len(work_orders)} work orders to SAP PM format: {output_file}"
        )
        return output_file

    def _map_work_type_to_sap(self, work_type: str) -> str:
        """Map work type to SAP PM order type."""
        mapping = {
            "inspection": "PM01",
            "repair": "PM02",
            "replacement": "PM03",
            "maintenance": "PM04",
        }
        return mapping.get(work_type.lower(), "PM04")

    def import_work_orders(self, source: str) -> List[Dict[str, Any]]:
        """Import work orders from SAP PM."""
        work_orders = []

        with open(source, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                work_orders.append(
                    {
                        "work_order_id": row.get("ORDERID"),
                        "pole_id": row.get("EQUIPMENT"),
                        "work_type": row.get("ORDER_TYPE"),
                        "status": row.get("STATUS", "new"),
                        "description": row.get("LONG_TEXT") or row.get("SHORT_TEXT"),
                        "scheduled_date": row.get("PLANNED_START"),
                        "priority": row.get("PRIORITY"),
                    }
                )

        return work_orders

    def export_pole_data(self, poles: List[PoleAssetExport]) -> str:
        """Export pole data to SAP PM equipment format."""
        sap_equipment = []

        for pole in poles:
            equipment = {
                "EQUIPMENT": pole.pole_id,
                "DESCRIPTION": f"Utility Pole {pole.pole_id}",
                "EQUIPMENT_TYPE": "POLE",
                "FUNCTION_LOC": f"POLE_{pole.pole_id}",
                "LATITUDE": pole.latitude,
                "LONGITUDE": pole.longitude,
                "INSTALL_DATE": (
                    pole.installation_date.strftime("%Y%m%d")
                    if pole.installation_date
                    else ""
                ),
                "MATERIAL": pole.material,
                "HEIGHT": pole.height_ft,
                "CLASS_RATING": pole.class_rating,
                "RISK_SCORE": pole.risk_score,
                "HEALTH_SCORE": pole.health_score,
                "LAST_INSPECTION": (
                    pole.last_inspection_date.strftime("%Y%m%d")
                    if pole.last_inspection_date
                    else ""
                ),
                "NEXT_INSPECTION": (
                    pole.next_inspection_due.strftime("%Y%m%d")
                    if pole.next_inspection_due
                    else ""
                ),
            }
            sap_equipment.append(equipment)

        output_file = (
            f"Output/sap_pm_equipment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        if sap_equipment:
            with open(output_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=sap_equipment[0].keys())
                writer.writeheader()
                writer.writerows(sap_equipment)

        return output_file


class ESRIAdapter(IntegrationAdapter):
    """Adapter for ESRI ArcGIS systems."""

    def __init__(self, geodatabase_path: Optional[str] = None):
        """Initialize ESRI adapter."""
        self.geodatabase_path = geodatabase_path

    def export_pole_data(self, poles: List[PoleAssetExport]) -> str:
        """Export pole data to ESRI GeoJSON/Shapefile format."""
        geojson_features = []

        for pole in poles:
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [pole.longitude, pole.latitude],
                },
                "properties": {
                    "pole_id": pole.pole_id,
                    "installation_date": (
                        pole.installation_date.isoformat()
                        if pole.installation_date
                        else None
                    ),
                    "material": pole.material,
                    "height_ft": pole.height_ft,
                    "class_rating": pole.class_rating,
                    "risk_score": pole.risk_score,
                    "health_score": pole.health_score,
                    "last_inspection": (
                        pole.last_inspection_date.isoformat()
                        if pole.last_inspection_date
                        else None
                    ),
                    "next_inspection": (
                        pole.next_inspection_due.isoformat()
                        if pole.next_inspection_due
                        else None
                    ),
                    "maintenance_status": pole.maintenance_status,
                },
            }
            geojson_features.append(feature)

        geojson = {
            "type": "FeatureCollection",
            "features": geojson_features,
            "crs": {
                "type": "name",
                "properties": {"name": "urn:ogc:def:crs:EPSG::4326"},
            },
        }

        output_file = (
            f"Output/esri_poles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.geojson"
        )
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(geojson, f, indent=2)

        logger.info(
            f"Exported {len(poles)} poles to ESRI GeoJSON format: {output_file}"
        )
        return output_file

    def import_pole_data(self, source: str) -> List[Dict[str, Any]]:
        """Import pole data from ESRI GeoJSON."""
        with open(source, "r") as f:
            geojson = json.load(f)

        poles = []
        for feature in geojson.get("features", []):
            props = feature.get("properties", {})
            coords = feature.get("geometry", {}).get("coordinates", [])

            poles.append(
                {
                    "pole_id": props.get("pole_id"),
                    "latitude": coords[1] if len(coords) >= 2 else None,
                    "longitude": coords[0] if len(coords) >= 1 else None,
                    "material": props.get("material"),
                    "height_ft": props.get("height_ft"),
                    "risk_score": props.get("risk_score"),
                    "health_score": props.get("health_score"),
                }
            )

        return poles

    def export_work_orders(self, work_orders: List[WorkOrderExport]) -> str:
        """Export work orders as ESRI point features."""
        features = []

        for wo in work_orders:
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [
                        wo.location.get("longitude"),
                        wo.location.get("latitude"),
                    ],
                },
                "properties": {
                    "work_order_id": wo.work_order_id,
                    "pole_id": wo.pole_id,
                    "work_type": wo.work_type,
                    "priority": wo.priority,
                    "description": wo.description,
                    "scheduled_date": wo.scheduled_date.isoformat(),
                    "estimated_duration": wo.estimated_duration_hours,
                    "status": wo.status,
                },
            }
            features.append(feature)

        geojson = {"type": "FeatureCollection", "features": features}

        output_file = (
            f"Output/esri_workorders_{datetime.now().strftime('%Y%m%d_%H%M%S')}.geojson"
        )
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(geojson, f, indent=2)

        return output_file


class OMSAdapter(IntegrationAdapter):
    """Adapter for Outage Management Systems."""

    def export_pole_data(self, poles: List[PoleAssetExport]) -> str:
        """Export pole data for outage management."""
        oms_data = []

        for pole in poles:
            oms_record = {
                "asset_id": pole.pole_id,
                "asset_type": "pole",
                "latitude": pole.latitude,
                "longitude": pole.longitude,
                "risk_score": pole.risk_score,
                "health_score": pole.health_score,
                "maintenance_status": pole.maintenance_status,
                "last_inspection": (
                    pole.last_inspection_date.isoformat()
                    if pole.last_inspection_date
                    else None
                ),
                "next_inspection": (
                    pole.next_inspection_due.isoformat()
                    if pole.next_inspection_due
                    else None
                ),
                "failure_probability": pole.risk_score * 0.1,  # Convert to probability
            }
            oms_data.append(oms_record)

        output_file = (
            f"Output/oms_poles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump({"assets": oms_data}, f, indent=2)

        return output_file

    def export_work_orders(self, work_orders: List[WorkOrderExport]) -> str:
        """Export planned maintenance for outage coordination."""
        oms_maintenance = []

        for wo in work_orders:
            maintenance = {
                "work_order_id": wo.work_order_id,
                "asset_id": wo.pole_id,
                "work_type": wo.work_type,
                "scheduled_start": wo.scheduled_date.isoformat(),
                "estimated_duration_hours": wo.estimated_duration_hours,
                "requires_outage": wo.work_type in ["repair", "replacement"],
                "customers_affected": 0,  # Would be populated from network model
                "coordinates": {
                    "latitude": wo.location.get("latitude"),
                    "longitude": wo.location.get("longitude"),
                },
            }
            oms_maintenance.append(maintenance)

        output_file = (
            f"Output/oms_maintenance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump({"planned_maintenance": oms_maintenance}, f, indent=2)

        return output_file


class IntegrationManager:
    """Manager for all integration adapters."""

    def __init__(self):
        """Initialize integration manager."""
        self.adapters: Dict[IntegrationSystem, IntegrationAdapter] = {}

    def register_adapter(self, system: IntegrationSystem, adapter: IntegrationAdapter):
        """Register an integration adapter."""
        self.adapters[system] = adapter

    def get_adapter(self, system: IntegrationSystem) -> Optional[IntegrationAdapter]:
        """Get an adapter for a system."""
        return self.adapters.get(system)

    def export_to_system(
        self, system: IntegrationSystem, data_type: str, data: List[Any]
    ) -> str:
        """Export data to a specific system."""
        adapter = self.get_adapter(system)
        if not adapter:
            raise ValueError(f"No adapter registered for {system}")

        if data_type == "work_orders":
            return adapter.export_work_orders(data)
        elif data_type == "pole_data":
            return adapter.export_pole_data(data)
        else:
            raise ValueError(f"Unknown data type: {data_type}")
