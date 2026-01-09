"""
Failure mode modeling for utility poles.

Utilities think in failure modes (rot, wind snap, vehicle strike, etc.) rather than
abstract risk scores. This module provides failure mode-specific analysis with
mode-specific predictors and inspection recommendations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

from .pole_data import PoleInfo, PoleHealthMetrics

logger = logging.getLogger(__name__)


class FailureMode(Enum):
    """Primary failure modes for utility poles."""

    ROT = "rot"  # Wood decay/rot
    GROUNDLINE_DECAY = "groundline_decay"  # Groundline rot (most common)
    WIND_SNAP = "wind_snap"  # Wind loading failure
    VEHICLE_STRIKE = "vehicle_strike"  # Collision damage
    FIRE = "fire"  # Wildfire or other fire damage
    HARDWARE_FAILURE = "hardware_failure"  # Crossarm, insulator, etc. failure
    ICE_LOAD = "ice_load"  # Ice accumulation failure
    FOUNDATION_FAILURE = "foundation_failure"  # Base/foundation issues


@dataclass
class FailureModeRisk:
    """Risk assessment for a specific failure mode."""

    mode: FailureMode
    probability: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    contributing_factors: Dict[str, float]  # Factor name -> contribution
    recommended_inspections: List[str]  # Inspection methods to use
    recommended_actions: List[str]  # Maintenance actions
    time_to_failure_estimate: Optional[float] = None  # Years
    time_to_failure_confidence: Optional[Tuple[float, float]] = (
        None  # (lower, upper) bounds
    )


@dataclass
class FailureModeAnalysis:
    """Complete failure mode analysis for a pole."""

    pole_id: str
    primary_mode: FailureMode  # Most likely failure mode
    mode_risks: List[FailureModeRisk]  # All assessed modes
    overall_risk: float  # Aggregate risk across all modes
    analysis_date: datetime = field(default_factory=datetime.now)
    model_version: str = "1.0.0"


class FailureModeModel:
    """Model for predicting failure modes with mode-specific predictors."""

    def __init__(self):
        """Initialize failure mode model with predictors and thresholds."""

        # Mode-specific predictors and weights
        self.mode_predictors = {
            FailureMode.ROT: {
                "soil_moisture": 0.35,
                "wood_type": 0.25,
                "age": 0.20,
                "groundline_condition": 0.20,
            },
            FailureMode.GROUNDLINE_DECAY: {
                "soil_moisture": 0.40,
                "groundline_depth": 0.25,
                "drainage": 0.20,
                "vegetation_contact": 0.15,
            },
            FailureMode.WIND_SNAP: {
                "wind_exposure": 0.30,
                "pole_strength_class": 0.25,
                "height": 0.20,
                "load_factor": 0.15,
                "age": 0.10,
            },
            FailureMode.VEHICLE_STRIKE: {
                "road_proximity": 0.40,
                "visibility": 0.25,
                "curvature": 0.20,
                "traffic_volume": 0.15,
            },
            FailureMode.FIRE: {
                "wildfire_risk_zone": 0.35,
                "vegetation_density": 0.25,
                "drought_index": 0.20,
                "fire_history": 0.20,
            },
            FailureMode.HARDWARE_FAILURE: {
                "hardware_age": 0.35,
                "hardware_type": 0.25,
                "load_factor": 0.20,
                "corrosion_exposure": 0.20,
            },
            FailureMode.ICE_LOAD: {
                "ice_zone": 0.35,
                "wind_exposure": 0.25,
                "pole_strength_class": 0.20,
                "conductor_configuration": 0.20,
            },
            FailureMode.FOUNDATION_FAILURE: {
                "soil_stability": 0.40,
                "foundation_type": 0.25,
                "erosion_risk": 0.20,
                "settlement_history": 0.15,
            },
        }

        # Base failure rates by mode (per year per pole)
        self.base_rates = {
            FailureMode.ROT: 0.002,
            FailureMode.GROUNDLINE_DECAY: 0.003,  # Most common
            FailureMode.WIND_SNAP: 0.0015,
            FailureMode.VEHICLE_STRIKE: 0.0008,
            FailureMode.FIRE: 0.0003,
            FailureMode.HARDWARE_FAILURE: 0.002,
            FailureMode.ICE_LOAD: 0.0005,
            FailureMode.FOUNDATION_FAILURE: 0.0005,
        }

    def assess_failure_modes(
        self,
        pole: PoleInfo,
        health_metrics: PoleHealthMetrics,
        environmental_data: Optional[Dict] = None,
    ) -> FailureModeAnalysis:
        """
        Assess all failure modes for a pole.

        Args:
            pole: Pole information
            health_metrics: Current health assessment metrics
            environmental_data: Optional environmental factors (weather, etc.)

        Returns:
            FailureModeAnalysis with all assessed modes
        """
        environmental_data = environmental_data or {}

        # Extract features for each mode
        features = self._extract_features(pole, health_metrics, environmental_data)

        # Assess each failure mode
        mode_risks = []
        for mode in FailureMode:
            risk = self._assess_single_mode(
                mode, pole, health_metrics, features, environmental_data
            )
            mode_risks.append(risk)

        # Sort by probability (highest first)
        mode_risks.sort(key=lambda x: x.probability, reverse=True)

        # Determine primary mode
        primary_mode = mode_risks[0].mode

        # Calculate overall aggregate risk (using independent event approximation)
        overall_risk = 1.0 - np.prod([1.0 - r.probability for r in mode_risks])

        return FailureModeAnalysis(
            pole_id=pole.pole_id,
            primary_mode=primary_mode,
            mode_risks=mode_risks,
            overall_risk=overall_risk,
        )

    def _extract_features(
        self,
        pole: PoleInfo,
        health_metrics: PoleHealthMetrics,
        environmental_data: Dict,
    ) -> Dict[str, float]:
        """Extract relevant features from pole and health data."""
        features = {}

        # Basic pole features
        age_years = getattr(pole, "age_years", 0) or 0
        features["age"] = min(age_years / 100.0, 1.0)  # Normalize (cap at 100 years)
        height_ft = getattr(pole, "height_ft", 40) or 40
        features["height"] = min(height_ft / 100.0, 1.0)  # Normalize
        features["wood_type"] = self._encode_wood_type(
            getattr(pole, "material", "unknown")
        )
        # Estimate strength class from height if not available
        features["pole_strength_class"] = min((height_ft - 30) / 40.0, 1.0)  # Normalize

        # Health metrics
        # Use moisture risk as proxy for soil moisture
        moisture_risk = getattr(health_metrics, "moisture_risk", 0.5)
        features["soil_moisture"] = moisture_risk  # Higher risk = higher moisture
        soil_stability = getattr(health_metrics, "soil_stability_score", 50) / 100.0
        features["groundline_condition"] = (
            1.0 - soil_stability
        )  # Lower score = worse condition
        structural_risk = getattr(health_metrics, "structural_risk_score", 50) / 100.0
        features["load_factor"] = structural_risk

        # Environmental features (with defaults)
        features["wind_exposure"] = environmental_data.get("wind_exposure", 0.5)
        features["road_proximity"] = environmental_data.get("road_proximity", 0.0)
        features["wildfire_risk_zone"] = environmental_data.get("wildfire_risk", 0.0)
        features["ice_zone"] = environmental_data.get("ice_zone", 0.0)
        features["drought_index"] = environmental_data.get("drought_index", 0.5)

        return features

    def _assess_single_mode(
        self,
        mode: FailureMode,
        pole: PoleInfo,
        health_metrics: PoleHealthMetrics,
        features: Dict[str, float],
        environmental_data: Dict,
    ) -> FailureModeRisk:
        """Assess risk for a single failure mode."""

        predictors = self.mode_predictors[mode]
        base_rate = self.base_rates[mode]

        # Calculate weighted risk score
        risk_score = 0.0
        contributing_factors = {}

        for factor, weight in predictors.items():
            factor_value = features.get(factor, 0.5)  # Default to middle
            contribution = factor_value * weight
            risk_score += contribution
            contributing_factors[factor] = contribution

        # Normalize and convert to probability (per year)
        # Scale base rate by risk score (0 = 0.1x base, 1 = 10x base)
        probability = base_rate * (0.1 + 0.9 * risk_score)
        probability = min(probability, 0.95)  # Cap at 95%

        # Calculate confidence (based on data completeness)
        confidence = self._calculate_confidence(mode, features, predictors)

        # Get recommended inspections
        recommended_inspections = self._get_inspection_methods(mode, risk_score)

        # Get recommended actions
        recommended_actions = self._get_recommended_actions(
            mode, risk_score, probability
        )

        # Estimate time to failure if high risk
        time_to_failure = None
        time_confidence = None
        if probability > 0.01:  # 1% annual probability
            time_to_failure, time_confidence = self._estimate_time_to_failure(
                probability
            )

        return FailureModeRisk(
            mode=mode,
            probability=probability,
            confidence=confidence,
            contributing_factors=contributing_factors,
            recommended_inspections=recommended_inspections,
            recommended_actions=recommended_actions,
            time_to_failure_estimate=time_to_failure,
            time_to_failure_confidence=time_confidence,
        )

    def _calculate_confidence(
        self, mode: FailureMode, features: Dict, predictors: Dict
    ) -> float:
        """Calculate confidence in the risk assessment."""
        # Lower confidence if key features are missing
        missing_weight = 0.0
        total_weight = 0.0

        for factor, weight in predictors.items():
            total_weight += weight
            if (
                features.get(factor) is None or features.get(factor) == 0.5
            ):  # Default value
                missing_weight += weight

        if total_weight == 0:
            return 0.5

        data_completeness = 1.0 - (missing_weight / total_weight)
        return max(0.3, data_completeness)  # Minimum 30% confidence

    def _get_inspection_methods(
        self, mode: FailureMode, risk_score: float
    ) -> List[str]:
        """Get recommended inspection methods for a failure mode."""
        base_methods = {
            FailureMode.ROT: ["visual", "sounding", "moisture_meter"],
            FailureMode.GROUNDLINE_DECAY: ["visual", "sounding", "resistograph"],
            FailureMode.WIND_SNAP: ["visual", "load_calculation", "strength_test"],
            FailureMode.VEHICLE_STRIKE: ["visual", "structural_assessment"],
            FailureMode.FIRE: ["visual", "vegetation_assessment"],
            FailureMode.HARDWARE_FAILURE: ["visual", "hardware_inspection"],
            FailureMode.ICE_LOAD: ["visual", "load_calculation", "ice_monitoring"],
            FailureMode.FOUNDATION_FAILURE: [
                "visual",
                "foundation_inspection",
                "soil_test",
            ],
        }

        methods = base_methods.get(mode, ["visual"])

        # Add advanced methods for high-risk poles
        if risk_score > 0.7:
            advanced_methods = {
                FailureMode.ROT: ["resistograph", "sonic_testing"],
                FailureMode.GROUNDLINE_DECAY: ["sonic_testing", "boring"],
                FailureMode.WIND_SNAP: ["load_rating", "engineering_review"],
                FailureMode.FOUNDATION_FAILURE: ["soil_analysis", "engineering_review"],
            }
            methods.extend(advanced_methods.get(mode, []))

        return methods

    def _get_recommended_actions(
        self, mode: FailureMode, risk_score: float, probability: float
    ) -> List[str]:
        """Get recommended maintenance actions for a failure mode."""
        actions = []

        if probability > 0.05:  # >5% annual probability
            actions.append("immediate_inspection")
            actions.append("structural_engineering_review")

        if risk_score > 0.8:
            actions.append("schedule_replacement")

        mode_specific_actions = {
            FailureMode.ROT: ["wood_preservative_treatment", "drainage_improvement"],
            FailureMode.GROUNDLINE_DECAY: ["groundline_treatment", "soil_drainage"],
            FailureMode.WIND_SNAP: ["guy_wire_installation", "load_reduction"],
            FailureMode.VEHICLE_STRIKE: ["barrier_installation", "signage"],
            FailureMode.FIRE: ["vegetation_clearance", "fire_retardant_treatment"],
            FailureMode.HARDWARE_FAILURE: [
                "hardware_replacement",
                "corrosion_protection",
            ],
            FailureMode.ICE_LOAD: ["load_management", "de_icing_equipment"],
            FailureMode.FOUNDATION_FAILURE: [
                "foundation_stabilization",
                "erosion_control",
            ],
        }

        actions.extend(mode_specific_actions.get(mode, []))
        return actions

    def _estimate_time_to_failure(
        self, annual_probability: float
    ) -> Tuple[float, Tuple[float, float]]:
        """
        Estimate time to failure using exponential distribution.

        Returns:
            (expected_years, (lower_bound, upper_bound))
        """
        # Mean time to failure = 1 / annual_rate
        expected_ttf = 1.0 / annual_probability if annual_probability > 0 else 100.0

        # Calculate confidence bounds (5th and 95th percentiles)
        # For exponential: lower = -ln(0.95)/rate, upper = -ln(0.05)/rate
        lower = (
            -np.log(0.95) / annual_probability
            if annual_probability > 0
            else expected_ttf * 0.5
        )
        upper = (
            -np.log(0.05) / annual_probability
            if annual_probability > 0
            else expected_ttf * 2.0
        )

        return expected_ttf, (lower, upper)

    def _encode_wood_type(self, wood_type: str) -> float:
        """Encode wood type to numeric value (higher = more susceptible to rot)."""
        encoding = {
            "southern_pine": 0.6,
            "douglas_fir": 0.4,
            "cedar": 0.3,
            "oak": 0.7,
            "unknown": 0.5,
        }
        return encoding.get(wood_type.lower(), 0.5)
