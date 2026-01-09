"""
Soil condition analysis and pole health assessment algorithms.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
from dataclasses import asdict

from .pole_data import PoleInfo, SoilSample, PoleHealthMetrics, WeatherData
from ..analysis.statistics import calculate_correlation, calculate_rmse

logger = logging.getLogger(__name__)

# Risk assessment constants
MOISTURE_RISK_SCALING_HIGH = 0.2  # Scaling factor for high moisture risk calculation
MOISTURE_RISK_SCALING_LOW = 0.05  # Scaling factor for low moisture risk calculation
MODERATE_RISK_OUTSIDE_OPTIMAL = (
    0.3  # Moderate risk for values outside optimal but acceptable
)
PH_RISK_SCALING_LOW = 2.0  # Scaling factor for low pH (acidic) risk
PH_RISK_SCALING_HIGH = 3.0  # Scaling factor for high pH (alkaline) risk
PH_RISK_MAX_ALKALINE = 0.7  # Maximum risk for alkaline soil
EC_RISK_SCALING = 4.0  # Scaling factor for electrical conductivity risk
BEARING_CAPACITY_RISK_SCALING = 50.0  # Scaling factor for bearing capacity risk
BEARING_CAPACITY_MODERATE_RISK = 0.4  # Moderate risk for suboptimal bearing capacity
DENSITY_RISK_SCALING = 0.5  # Scaling factor for bulk density risk
DENSITY_RISK_MAX_LOW = 0.8  # Maximum risk for too loose (low density)
DENSITY_RISK_MAX_HIGH = 0.6  # Maximum risk for too dense
TREND_SLOPE_THRESHOLD = 0.01  # Significant change threshold per sample
TREND_RISK_SCALING = 10.0  # Scaling factor for trend risk
NETWORK_DENSITY_THRESHOLD = 0.8  # High density threshold for network analysis
NETWORK_DENSITY_RISK_INCREMENT = 0.3  # Risk increment for high network density
NETWORK_ANOMALY_RISK_INCREMENT = 0.2  # Risk increment for network anomalies
MIN_SAMPLES_FOR_TREND = 2  # Minimum samples required for trend analysis
MIN_SAMPLES_FOR_NETWORK = 10  # Minimum samples required for network analysis


class SoilConditionAnalyzer:
    """Analyzes soil conditions for utility pole stability assessment."""

    def __init__(self):
        # Optimal ranges for different soil parameters
        self.optimal_ranges = {
            "moisture_content": (0.15, 0.35),  # m³/m³
            "ph": (6.0, 8.0),
            "bulk_density": (1.2, 1.8),  # g/cm³
            "electrical_conductivity": (0.0, 2.0),  # dS/m
            "bearing_capacity": (100, 500),  # kPa
            "porosity": (0.35, 0.55),  # fraction
        }

        # Critical thresholds for risk assessment
        self.critical_thresholds = {
            "moisture_content_high": 0.45,  # High moisture - erosion/instability risk
            "moisture_content_low": 0.08,  # Too dry - poor compaction
            "ph_low": 5.0,  # Acidic - corrosion risk
            "ph_high": 9.0,  # Alkaline - chemical reactions
            "electrical_conductivity_high": 4.0,  # High salinity
            "bearing_capacity_low": 50,  # kPa - insufficient support
        }

    def analyze_soil_sample(self, sample: SoilSample) -> Dict[str, float]:
        """
        Analyze a single soil sample and return risk scores.

        Returns:
            Dictionary with risk scores (0-1, higher is worse)
        """
        risks = {}

        # Moisture content analysis
        if sample.moisture_content is not None:
            moisture_risk = self._assess_moisture_risk(sample.moisture_content)
            risks["moisture_risk"] = moisture_risk

        # pH analysis
        if sample.ph is not None:
            ph_risk = self._assess_ph_risk(sample.ph)
            risks["chemical_corrosion_risk"] = ph_risk

        # Electrical conductivity (salinity)
        if sample.electrical_conductivity is not None:
            ec_risk = self._assess_ec_risk(sample.electrical_conductivity)
            risks["salinity_risk"] = ec_risk

        # Bearing capacity
        if sample.bearing_capacity is not None:
            bearing_risk = self._assess_bearing_capacity_risk(sample.bearing_capacity)
            risks["bearing_capacity_risk"] = bearing_risk

        # Bulk density
        if sample.bulk_density is not None:
            density_risk = self._assess_density_risk(sample.bulk_density)
            risks["compaction_risk"] = density_risk

        return risks

    def _assess_moisture_risk(self, moisture: float) -> float:
        """Assess risk from soil moisture content."""
        high_threshold = self.critical_thresholds["moisture_content_high"]
        low_threshold = self.critical_thresholds["moisture_content_low"]

        if moisture > high_threshold:
            # Very high moisture - erosion, instability
            return min(
                1.0,
                (moisture - high_threshold) / MOISTURE_RISK_SCALING_HIGH,
            )
        elif moisture < low_threshold:
            # Very low moisture - poor compaction, dust
            return min(
                1.0,
                (low_threshold - moisture) / MOISTURE_RISK_SCALING_LOW,
            )
        else:
            # Within acceptable range
            optimal_min, optimal_max = self.optimal_ranges["moisture_content"]
            if optimal_min <= moisture <= optimal_max:
                return 0.0
            else:
                # Moderate risk outside optimal range but within acceptable limits
                return MODERATE_RISK_OUTSIDE_OPTIMAL

    def _assess_ph_risk(self, ph: float) -> float:
        """Assess corrosion risk from soil pH."""
        ph_low = self.critical_thresholds["ph_low"]
        ph_high = self.critical_thresholds["ph_high"]

        if ph < ph_low:
            # Acidic soil - high corrosion risk
            return min(1.0, (ph_low - ph) / PH_RISK_SCALING_LOW)
        elif ph > ph_high:
            # Alkaline soil - moderate chemical reaction risk
            return min(PH_RISK_MAX_ALKALINE, (ph - ph_high) / PH_RISK_SCALING_HIGH)
        else:
            return 0.0

    def _assess_ec_risk(self, ec: float) -> float:
        """Assess risk from electrical conductivity (salinity)."""
        ec_high = self.critical_thresholds["electrical_conductivity_high"]

        if ec > ec_high:
            # High salinity - corrosion risk
            return min(
                1.0,
                (ec - ec_high) / EC_RISK_SCALING,
            )
        else:
            return 0.0

    def _assess_bearing_capacity_risk(self, bearing_capacity: float) -> float:
        """Assess structural support risk from bearing capacity."""
        capacity_low = self.critical_thresholds["bearing_capacity_low"]

        if bearing_capacity < capacity_low:
            # Insufficient bearing capacity
            return min(
                1.0,
                (capacity_low - bearing_capacity) / BEARING_CAPACITY_RISK_SCALING,
            )
        else:
            optimal_min, optimal_max = self.optimal_ranges["bearing_capacity"]
            if bearing_capacity < optimal_min:
                return BEARING_CAPACITY_MODERATE_RISK
            return 0.0

    def _assess_density_risk(self, bulk_density: float) -> float:
        """Assess compaction risk from bulk density."""
        optimal_min, optimal_max = self.optimal_ranges["bulk_density"]

        if bulk_density < optimal_min:
            # Too loose - poor support
            return min(
                DENSITY_RISK_MAX_LOW,
                (optimal_min - bulk_density) / DENSITY_RISK_SCALING,
            )
        elif bulk_density > optimal_max:
            # Too dense - drainage issues
            return min(
                DENSITY_RISK_MAX_HIGH,
                (bulk_density - optimal_max) / DENSITY_RISK_SCALING,
            )
        return 0.0

    def analyze_temporal_trends(
        self, samples: List[SoilSample], use_network_analysis: bool = False
    ) -> Dict[str, float]:
        """
        Analyze trends in soil conditions over time.

        Args:
            samples: List of soil samples
            use_network_analysis: If True, use ts2net for network-based pattern detection

        Returns:
            Dictionary containing trend analysis results including trend_risk,
            moisture_trend, and optional network analysis features.
        """
        if len(samples) < MIN_SAMPLES_FOR_TREND:
            return {"trend_risk": 0.0}

        # Sort samples by date
        sorted_samples = sorted(samples, key=lambda x: x.sample_date)

        # Extract time series data - use list comprehension with filter (faster)
        # Pre-allocate numpy array instead of list.append()
        moisture_values = np.array(
            [
                s.moisture_content
                for s in sorted_samples
                if s.moisture_content is not None
            ],
            dtype=np.float64,
        )
        dates = np.array(
            [s.sample_date for s in sorted_samples[: len(moisture_values)]]
        )

        if len(moisture_values) < MIN_SAMPLES_FOR_TREND:
            return {"trend_risk": 0.0}

        # Calculate basic trend using vectorized operations
        x = np.arange(len(moisture_values), dtype=np.float64)
        slope = np.polyfit(x, moisture_values, 1)[0]

        # Assess trend risk
        trend_risk = 0.0
        if abs(slope) > TREND_SLOPE_THRESHOLD:
            trend_risk = min(1.0, abs(slope) * TREND_RISK_SCALING)

        result = {
            "trend_risk": trend_risk,
            "moisture_trend": slope,
            "trend_direction": "increasing" if slope > 0 else "decreasing",
        }

        # Add network-based analysis if requested and available
        if use_network_analysis and len(moisture_values) >= MIN_SAMPLES_FOR_NETWORK:
            try:
                from .ts2net_integration import TS2NetAnalyzer, NetworkMethod
                import pandas as pd

                analyzer = TS2NetAnalyzer()
                timestamps = pd.DatetimeIndex(dates[: len(moisture_values)])

                network_analysis = analyzer.analyze_time_series(
                    pole_id="",  # Not needed for trend analysis
                    time_series=np.array(moisture_values),
                    time_series_name="soil_moisture",
                    timestamps=timestamps,
                    methods=[NetworkMethod.HVG, NetworkMethod.TRANSITION],
                )

                if network_analysis and network_analysis.hvg_features:
                    result["network_density"] = network_analysis.hvg_features.density
                    result["network_avg_degree"] = (
                        network_analysis.hvg_features.avg_degree
                    )
                    result["network_complexity"] = network_analysis.complexity_score
                    result["network_entropy"] = (
                        network_analysis.hvg_features.degree_entropy
                    )

                    # High density may indicate complex patterns (potential issues)
                    if (
                        network_analysis.hvg_features.density
                        > NETWORK_DENSITY_THRESHOLD
                    ):
                        result["trend_risk"] = max(
                            result["trend_risk"], NETWORK_DENSITY_RISK_INCREMENT
                        )

                    if network_analysis.is_anomalous:
                        result["network_anomaly"] = True
                        result["network_anomaly_score"] = network_analysis.anomaly_score
                        result["network_anomaly_reasons"] = (
                            network_analysis.anomaly_reasons
                        )
                        result["trend_risk"] = min(
                            1.0, result["trend_risk"] + NETWORK_ANOMALY_RISK_INCREMENT
                        )

                # Add transition network features for pattern detection
                if network_analysis and network_analysis.transition_features:
                    result["transition_edges"] = (
                        network_analysis.transition_features.n_edges
                    )
                    if network_analysis.is_periodic:
                        result["periodic_pattern"] = True
                    if network_analysis.is_chaotic:
                        result["chaotic_pattern"] = True
            except ImportError:
                logger.debug("ts2net not available for network analysis")
            except Exception as e:
                logger.warning(f"Network analysis failed: {e}")

        # Add comprehensive anomaly detection using anomaly-detection-toolkit
        if len(moisture_values) >= MIN_SAMPLES_FOR_NETWORK:
            try:
                from .anomaly_detection_integration import (
                    AnomalyDetector,
                    AnomalyDetectionMethod,
                    EnsembleAnomalyDetector,
                )
                import pandas as pd

                # Use ensemble detection for robust results
                ensemble_detector = EnsembleAnomalyDetector(
                    methods=[
                        AnomalyDetectionMethod.ISOLATION_FOREST,
                        AnomalyDetectionMethod.LOF,
                        AnomalyDetectionMethod.WAVELET,
                    ],
                    method_params={
                        "contamination": 0.05,  # Expect 5% anomalies
                    },
                )

                timestamps = pd.DatetimeIndex(dates[: len(moisture_values)])
                ensemble_result = ensemble_detector.detect_ensemble(
                    time_series=moisture_values,
                    timestamps=timestamps,
                    pole_id="",
                    time_series_name="soil_moisture",
                    consensus_threshold=0.5,  # 50% of detectors must agree
                )

                # Add anomaly detection results
                result["anomaly_detection_n_anomalies"] = ensemble_result.n_anomalies
                result["anomaly_detection_rate"] = ensemble_result.anomaly_rate
                result["anomaly_detection_mean_score"] = float(
                    ensemble_result.consensus_scores.mean()
                )
                result["anomaly_detection_max_score"] = float(
                    ensemble_result.consensus_scores.max()
                )

                # Individual detector results
                for (
                    method_name,
                    individual_result,
                ) in ensemble_result.individual_results.items():
                    result[f"anomaly_{method_name}_n_anomalies"] = (
                        individual_result.n_anomalies
                    )
                    result[f"anomaly_{method_name}_rate"] = (
                        individual_result.anomaly_rate
                    )

                # Flag if anomalies detected by consensus
                if ensemble_result.n_anomalies > 0:
                    result["anomaly_detected"] = True
                    result["trend_risk"] = min(
                        1.0,
                        result["trend_risk"]
                        + (
                            ensemble_result.anomaly_rate / 100 * 0.3
                        ),  # Up to 0.3 risk increment
                    )
                else:
                    result["anomaly_detected"] = False

            except ImportError:
                logger.debug(
                    "anomaly-detection-toolkit not available for anomaly detection"
                )
            except Exception as e:
                logger.warning(f"Anomaly detection failed: {e}")

        return result


class PoleHealthAssessment:
    """Main class for assessing utility pole health based on soil and pole conditions."""

    def __init__(self):
        self.soil_analyzer = SoilConditionAnalyzer()

        # Pole-specific risk factors
        self.pole_risk_factors = {
            "wood": {
                "age_threshold_years": 25,
                "moisture_sensitivity": 1.2,
                "ph_sensitivity": 1.5,
                "base_failure_rate": 0.02,
            },
            "concrete": {
                "age_threshold_years": 40,
                "moisture_sensitivity": 0.8,
                "ph_sensitivity": 1.0,
                "base_failure_rate": 0.01,
            },
            "steel": {
                "age_threshold_years": 50,
                "moisture_sensitivity": 1.0,
                "ph_sensitivity": 2.0,  # High corrosion sensitivity
                "base_failure_rate": 0.015,
            },
            "composite": {
                "age_threshold_years": 60,
                "moisture_sensitivity": 0.6,
                "ph_sensitivity": 0.5,
                "base_failure_rate": 0.005,
            },
        }

    def assess_pole_health(
        self,
        pole: PoleInfo,
        soil_samples: List[SoilSample],
        weather_data: Optional[List[WeatherData]] = None,
    ) -> PoleHealthMetrics:
        """
        Perform comprehensive pole health assessment.

        Args:
            pole: Pole information
            soil_samples: List of soil samples for this pole
            weather_data: Optional weather data for enhanced assessment

        Returns:
            PoleHealthMetrics with assessment results
        """
        if not soil_samples:
            logger.warning(f"No soil samples available for pole {pole.pole_id}")
            return self._create_default_metrics(pole)

        # Get latest soil sample
        latest_sample = max(soil_samples, key=lambda x: x.sample_date)

        # Analyze soil conditions
        soil_risks = self.soil_analyzer.analyze_soil_sample(latest_sample)

        # Analyze temporal trends if multiple samples available
        temporal_analysis = self.soil_analyzer.analyze_temporal_trends(soil_samples)
        soil_risks.update(temporal_analysis)

        # Assess pole-specific risks
        pole_risks = self._assess_pole_specific_risks(pole, latest_sample)

        # Calculate overall scores
        overall_health_score = self._calculate_overall_health_score(
            soil_risks, pole_risks
        )
        soil_stability_score = self._calculate_soil_stability_score(soil_risks)
        structural_risk_score = self._calculate_structural_risk_score(pole_risks)

        # Determine maintenance priority
        maintenance_priority = self._determine_maintenance_priority(
            overall_health_score, soil_risks, pole_risks
        )

        # Create health metrics
        metrics = PoleHealthMetrics(
            pole_id=pole.pole_id,
            assessment_date=datetime.now(),
            overall_health_score=overall_health_score,
            soil_stability_score=soil_stability_score,
            structural_risk_score=structural_risk_score,
            moisture_risk=soil_risks.get("moisture_risk", 0.0),
            erosion_risk=self._calculate_erosion_risk(soil_risks, weather_data),
            chemical_corrosion_risk=soil_risks.get("chemical_corrosion_risk", 0.0),
            freeze_thaw_risk=self._calculate_freeze_thaw_risk(soil_risks, weather_data),
            bearing_capacity_risk=soil_risks.get("bearing_capacity_risk", 0.0),
            maintenance_priority=maintenance_priority,
            confidence_level=self._calculate_confidence_level(soil_samples, pole),
            data_completeness=self._calculate_data_completeness(latest_sample),
        )

        # Set action flags
        metrics.requires_immediate_attention = (
            overall_health_score < 30 or maintenance_priority == "critical"
        )
        metrics.requires_monitoring = overall_health_score < 60

        return metrics

    def _assess_pole_specific_risks(
        self, pole: PoleInfo, soil_sample: SoilSample
    ) -> Dict[str, float]:
        """Assess risks specific to pole type and characteristics."""
        pole_type = pole.pole_type.lower()
        if pole_type not in self.pole_risk_factors:
            pole_type = "wood"  # Default

        factors = self.pole_risk_factors[pole_type]
        risks = {}

        # Age-related risk
        if pole.age_years:
            age_risk = max(0.0, (pole.age_years - factors["age_threshold_years"]) / 20)
            risks["age_risk"] = min(1.0, age_risk)
        else:
            risks["age_risk"] = 0.3  # Unknown age = moderate risk

        # Material-specific moisture sensitivity
        if soil_sample.moisture_content is not None:
            base_moisture_risk = self.soil_analyzer._assess_moisture_risk(
                soil_sample.moisture_content
            )
            risks["material_moisture_risk"] = (
                base_moisture_risk * factors["moisture_sensitivity"]
            )

        # Material-specific pH sensitivity
        if soil_sample.ph is not None:
            base_ph_risk = self.soil_analyzer._assess_ph_risk(soil_sample.ph)
            risks["material_ph_risk"] = base_ph_risk * factors["ph_sensitivity"]

        return risks

    def _calculate_overall_health_score(
        self, soil_risks: Dict[str, float], pole_risks: Dict[str, float]
    ) -> float:
        """Calculate overall health score (0-100, higher is better)."""
        # Combine all risk factors
        all_risks = list(soil_risks.values()) + list(pole_risks.values())
        if not all_risks:
            return 50.0  # Neutral score if no data

        # Weight different risk factors
        weighted_risks = []

        # High-priority risks
        for risk_name in [
            "moisture_risk",
            "bearing_capacity_risk",
            "chemical_corrosion_risk",
        ]:
            if risk_name in soil_risks:
                weighted_risks.extend([soil_risks[risk_name]] * 2)  # Double weight

        # Medium-priority risks
        for risk_name in ["age_risk", "material_moisture_risk", "material_ph_risk"]:
            if risk_name in pole_risks:
                weighted_risks.append(pole_risks[risk_name])

        # Add remaining risks with lower weight
        for risk in all_risks:
            if isinstance(risk, (int, float)):
                weighted_risks.append(risk * 0.5)

        # Calculate average risk and convert to health score
        if weighted_risks:
            avg_risk = np.mean(weighted_risks)
            health_score = max(0, min(100, (1 - avg_risk) * 100))
        else:
            health_score = 50.0  # Default neutral score

        return health_score

    def _calculate_soil_stability_score(self, soil_risks: Dict[str, float]) -> float:
        """Calculate soil stability score (0-100, higher is better)."""
        stability_risks = [
            soil_risks.get("moisture_risk", 0.0),
            soil_risks.get("bearing_capacity_risk", 0.0),
            soil_risks.get("compaction_risk", 0.0),
        ]
        # Filter out any non-numeric values
        stability_risks = [r for r in stability_risks if isinstance(r, (int, float))]
        if stability_risks:
            avg_risk = np.mean(stability_risks)
        else:
            avg_risk = 0.0
        return max(0, min(100, (1 - avg_risk) * 100))

    def _calculate_structural_risk_score(self, pole_risks: Dict[str, float]) -> float:
        """Calculate structural risk score (0-100, higher is worse)."""
        structural_risks = [
            pole_risks.get("age_risk", 0.0),
            pole_risks.get("material_moisture_risk", 0.0),
            pole_risks.get("material_ph_risk", 0.0),
        ]
        # Filter out any non-numeric values
        structural_risks = [r for r in structural_risks if isinstance(r, (int, float))]
        if structural_risks:
            avg_risk = np.mean(structural_risks)
        else:
            avg_risk = 0.0
        return max(0, min(100, avg_risk * 100))

    def _calculate_erosion_risk(
        self, soil_risks: Dict[str, float], weather_data: Optional[List[WeatherData]]
    ) -> float:
        """Calculate erosion risk based on soil and weather conditions."""
        base_erosion_risk = soil_risks.get("moisture_risk", 0.0) * 0.5

        if weather_data:
            # Factor in precipitation intensity
            recent_weather = sorted(weather_data, key=lambda x: x.date)[
                -30:
            ]  # Last 30 days
            heavy_rain_days = sum(
                1 for w in recent_weather if w.precipitation_intensity == "heavy"
            )

            if heavy_rain_days > 5:
                base_erosion_risk *= 1.5

        return min(1.0, base_erosion_risk)

    def _calculate_freeze_thaw_risk(
        self, soil_risks: Dict[str, float], weather_data: Optional[List[WeatherData]]
    ) -> float:
        """Calculate freeze-thaw risk."""
        if not weather_data:
            return 0.2  # Default moderate risk

        # Count freeze-thaw cycles
        total_cycles = sum(
            w.freeze_thaw_cycles_count or 0
            for w in weather_data
            if w.freeze_thaw_cycles_count
        )
        if total_cycles > 50:  # High number of cycles
            return min(1.0, total_cycles / 100)

        return min(0.5, total_cycles / 100)

    def _determine_maintenance_priority(
        self,
        health_score: float,
        soil_risks: Dict[str, float],
        pole_risks: Dict[str, float],
    ) -> str:
        """Determine maintenance priority level."""
        if health_score < 20:
            return "critical"
        elif health_score < 40:
            return "high"
        elif health_score < 70:
            return "medium"
        else:
            return "low"

    def _calculate_confidence_level(
        self, soil_samples: List[SoilSample], pole: PoleInfo
    ) -> float:
        """Calculate confidence level in the assessment."""
        confidence = 0.5  # Base confidence

        # More samples = higher confidence
        if len(soil_samples) >= 3:
            confidence += 0.2
        elif len(soil_samples) >= 2:
            confidence += 0.1

        # Recent samples = higher confidence
        latest_sample = max(soil_samples, key=lambda x: x.sample_date)
        days_since_sample = (datetime.now() - latest_sample.sample_date).days
        if days_since_sample < 30:
            confidence += 0.2
        elif days_since_sample < 90:
            confidence += 0.1

        # Complete pole info = higher confidence
        if all([pole.height_ft, pole.install_date, pole.material]):
            confidence += 0.1

        return min(1.0, confidence)

    def _calculate_data_completeness(self, sample: SoilSample) -> float:
        """Calculate completeness of available data."""
        required_fields = ["moisture_content", "ph", "bulk_density", "bearing_capacity"]
        available_fields = sum(
            1 for field in required_fields if getattr(sample, field) is not None
        )
        return available_fields / len(required_fields)

    def _create_default_metrics(self, pole: PoleInfo) -> PoleHealthMetrics:
        """Create default metrics when no soil data is available."""
        return PoleHealthMetrics(
            pole_id=pole.pole_id,
            assessment_date=datetime.now(),
            overall_health_score=50.0,  # Neutral score
            soil_stability_score=50.0,
            structural_risk_score=50.0,
            moisture_risk=0.5,
            erosion_risk=0.5,
            chemical_corrosion_risk=0.5,
            freeze_thaw_risk=0.5,
            bearing_capacity_risk=0.5,
            maintenance_priority="medium",
            confidence_level=0.1,  # Very low confidence without soil data
            data_completeness=0.0,
        )
