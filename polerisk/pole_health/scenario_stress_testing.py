"""
Scenario stress testing for utility poles.

Allows users to simulate hurricanes, ice storms, wildfire seasons, load growth,
and other scenarios. Shifts weather inputs and recomputes outages, costs, and
crew demand. Essential for planners before rate cases.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ScenarioType(Enum):
    """Types of stress test scenarios."""

    HURRICANE = "hurricane"
    ICE_STORM = "ice_storm"
    WILDFIRE_SEASON = "wildfire_season"
    LOAD_GROWTH = "load_growth"
    DROUGHT = "drought"
    EXTREME_HEAT = "extreme_heat"
    WIND_EVENT = "wind_event"
    EARTHQUAKE = "earthquake"


@dataclass
class ScenarioParameters:
    """Parameters for a stress test scenario."""

    scenario_type: ScenarioType
    severity: str  # "moderate", "severe", "extreme"
    duration_days: int
    affected_region: Optional[str] = None
    wind_speed_mph: Optional[float] = None
    ice_thickness_inches: Optional[float] = None
    temperature_f: Optional[float] = None
    load_increase_percent: Optional[float] = None
    custom_parameters: Dict = field(default_factory=dict)


@dataclass
class StressTestResult:
    """Results from a stress test scenario."""

    scenario_type: ScenarioType
    total_poles_affected: int
    poles_at_risk: int  # High risk of failure
    expected_failures: float  # Expected number of failures
    expected_outage_hours: float
    expected_customers_affected: int
    estimated_cost: float  # Total cost including repairs, outages, etc.
    crew_demand_person_days: float
    material_demand: Dict[str, float]  # Material type -> quantity needed
    time_to_restore_hours: float
    risk_distribution: Dict[str, int]  # Risk level -> count of poles


class ScenarioStressTester:
    """Model for stress testing scenarios."""

    def __init__(self):
        """Initialize stress testing model."""

        # Scenario severity multipliers
        self.severity_multipliers = {
            "moderate": 1.5,
            "severe": 2.5,
            "extreme": 4.0,
        }

        # Base failure rates by scenario type
        self.base_failure_rates = {
            ScenarioType.HURRICANE: 0.02,  # 2% base failure rate
            ScenarioType.ICE_STORM: 0.015,
            ScenarioType.WILDFIRE_SEASON: 0.01,
            ScenarioType.LOAD_GROWTH: 0.005,
            ScenarioType.DROUGHT: 0.008,
            ScenarioType.EXTREME_HEAT: 0.006,
            ScenarioType.WIND_EVENT: 0.012,
            ScenarioType.EARTHQUAKE: 0.03,
        }

    def run_stress_test(
        self,
        poles_df: pd.DataFrame,
        scenario: ScenarioParameters,
        base_risk_scores: Optional[pd.Series] = None,
    ) -> StressTestResult:
        """
        Run a stress test scenario on a fleet of poles.

        Args:
            poles_df: DataFrame with pole information
            scenario: Scenario parameters
            base_risk_scores: Optional pre-calculated risk scores

        Returns:
            StressTestResult with scenario impacts
        """
        # Calculate scenario-adjusted risk scores
        adjusted_risks = self._calculate_adjusted_risks(
            poles_df, scenario, base_risk_scores
        )

        # Identify poles at risk
        high_risk_threshold = 0.7
        poles_at_risk = (adjusted_risks > high_risk_threshold).sum()

        # Calculate expected failures
        expected_failures = self._estimate_failures(poles_df, adjusted_risks, scenario)

        # Estimate outage impacts
        outage_hours, customers_affected = self._estimate_outage_impacts(
            poles_df, expected_failures, scenario
        )

        # Estimate costs
        estimated_cost = self._estimate_costs(poles_df, expected_failures, scenario)

        # Estimate crew and material demand
        crew_demand, material_demand = self._estimate_resource_demand(
            poles_df, expected_failures, scenario
        )

        # Estimate time to restore
        time_to_restore = self._estimate_restoration_time(
            expected_failures, crew_demand, scenario
        )

        # Risk distribution
        risk_distribution = self._calculate_risk_distribution(adjusted_risks)

        return StressTestResult(
            scenario_type=scenario.scenario_type,
            total_poles_affected=len(poles_df),
            poles_at_risk=poles_at_risk,
            expected_failures=expected_failures,
            expected_outage_hours=outage_hours,
            expected_customers_affected=customers_affected,
            estimated_cost=estimated_cost,
            crew_demand_person_days=crew_demand,
            material_demand=material_demand,
            time_to_restore_hours=time_to_restore,
            risk_distribution=risk_distribution,
        )

    def compare_scenarios(
        self, poles_df: pd.DataFrame, scenarios: List[ScenarioParameters]
    ) -> pd.DataFrame:
        """
        Compare multiple scenarios side-by-side.

        Args:
            poles_df: DataFrame with pole information
            scenarios: List of scenarios to compare

        Returns:
            DataFrame with scenario comparison results
        """
        results = []

        for scenario in scenarios:
            result = self.run_stress_test(poles_df, scenario)
            results.append(
                {
                    "scenario_type": scenario.scenario_type.value,
                    "severity": scenario.severity,
                    "expected_failures": result.expected_failures,
                    "expected_outage_hours": result.expected_outage_hours,
                    "expected_customers_affected": result.expected_customers_affected,
                    "estimated_cost": result.estimated_cost,
                    "crew_demand_person_days": result.crew_demand_person_days,
                    "time_to_restore_hours": result.time_to_restore_hours,
                    "poles_at_risk": result.poles_at_risk,
                }
            )

        return pd.DataFrame(results)

    def _calculate_adjusted_risks(
        self,
        poles_df: pd.DataFrame,
        scenario: ScenarioParameters,
        base_risk_scores: Optional[pd.Series],
    ) -> pd.Series:
        """Calculate scenario-adjusted risk scores."""
        if base_risk_scores is None:
            # Assume baseline risk of 0.3 if not provided
            base_risks = pd.Series(0.3, index=poles_df.index)
        else:
            base_risks = base_risk_scores.copy()

        severity_mult = self.severity_multipliers.get(scenario.severity, 1.0)
        base_rate = self.base_failure_rates.get(scenario.scenario_type, 0.01)

        # Scenario-specific adjustments
        if scenario.scenario_type == ScenarioType.HURRICANE:
            # Higher risk for taller poles, coastal areas
            if "height_ft" in poles_df.columns:
                height_factor = (poles_df["height_ft"] / 50.0).clip(0.5, 2.0)
            else:
                height_factor = 1.0
            adjustment = base_rate * severity_mult * height_factor
        elif scenario.scenario_type == ScenarioType.ICE_STORM:
            # Higher risk for loaded poles
            adjustment = base_rate * severity_mult * 1.2
        elif scenario.scenario_type == ScenarioType.WILDFIRE_SEASON:
            # Higher risk in forested/vegetated areas
            adjustment = base_rate * severity_mult
        elif scenario.scenario_type == ScenarioType.LOAD_GROWTH:
            # Incremental risk increase
            load_increase = scenario.load_increase_percent or 20.0
            adjustment = base_rate * (load_increase / 100.0)
        else:
            adjustment = base_rate * severity_mult

        # Apply adjustment to base risks
        adjusted_risks = base_risks + adjustment
        adjusted_risks = adjusted_risks.clip(0.0, 1.0)

        return adjusted_risks

    def _estimate_failures(
        self,
        poles_df: pd.DataFrame,
        adjusted_risks: pd.Series,
        scenario: ScenarioParameters,
    ) -> float:
        """Estimate expected number of failures."""
        # Expected failures = sum of failure probabilities
        expected_failures = adjusted_risks.sum()
        return float(expected_failures)

    def _estimate_outage_impacts(
        self,
        poles_df: pd.DataFrame,
        expected_failures: float,
        scenario: ScenarioParameters,
    ) -> Tuple[float, int]:
        """Estimate outage hours and customers affected."""
        # Average outage duration per failure (hours)
        outage_durations = {
            ScenarioType.HURRICANE: 72.0,
            ScenarioType.ICE_STORM: 48.0,
            ScenarioType.WILDFIRE_SEASON: 24.0,
            ScenarioType.LOAD_GROWTH: 12.0,
            ScenarioType.DROUGHT: 8.0,
            ScenarioType.EXTREME_HEAT: 6.0,
            ScenarioType.WIND_EVENT: 18.0,
            ScenarioType.EARTHQUAKE: 96.0,
        }

        avg_outage_hours = outage_durations.get(scenario.scenario_type, 24.0)
        total_outage_hours = expected_failures * avg_outage_hours

        # Estimate customers affected (assuming poles serve different numbers)
        if "customers_served" in poles_df.columns:
            # Weighted by customer count
            customers_per_failure = poles_df["customers_served"].mean()
        else:
            customers_per_failure = 50  # Default assumption

        customers_affected = int(expected_failures * customers_per_failure)

        return total_outage_hours, customers_affected

    def _estimate_costs(
        self,
        poles_df: pd.DataFrame,
        expected_failures: float,
        scenario: ScenarioParameters,
    ) -> float:
        """Estimate total scenario costs."""
        # Replacement cost per pole
        replacement_cost = 8000.0  # $8k per pole on average

        # Outage cost per customer-hour
        outage_cost_per_customer_hour = 10.0

        # Calculate costs
        replacement_costs = expected_failures * replacement_cost

        # Outage costs (simplified)
        _, customers_affected = self._estimate_outage_impacts(
            poles_df, expected_failures, scenario
        )
        outage_costs = customers_affected * 50.0  # Simplified outage cost

        # Emergency response costs
        emergency_multiplier = self.severity_multipliers.get(scenario.severity, 1.0)
        emergency_costs = expected_failures * 2000.0 * emergency_multiplier

        total_cost = replacement_costs + outage_costs + emergency_costs

        return total_cost

    def _estimate_resource_demand(
        self,
        poles_df: pd.DataFrame,
        expected_failures: float,
        scenario: ScenarioParameters,
    ) -> Tuple[float, Dict[str, float]]:
        """Estimate crew and material demand."""
        # Crew demand (person-days per failure)
        crew_per_failure = {
            ScenarioType.HURRICANE: 2.5,
            ScenarioType.ICE_STORM: 3.0,
            ScenarioType.WILDFIRE_SEASON: 2.0,
            ScenarioType.LOAD_GROWTH: 1.5,
            ScenarioType.DROUGHT: 1.0,
            ScenarioType.EXTREME_HEAT: 1.0,
            ScenarioType.WIND_EVENT: 2.0,
            ScenarioType.EARTHQUAKE: 4.0,
        }

        crew_per_fail = crew_per_failure.get(scenario.scenario_type, 2.0)
        total_crew_demand = expected_failures * crew_per_fail

        # Material demand
        material_demand = {
            "poles": expected_failures,
            "hardware": expected_failures * 500.0,  # $500 in hardware per pole
            "concrete": expected_failures * 0.5,  # cubic yards
            "equipment_rental_days": expected_failures * 2.0,
        }

        return total_crew_demand, material_demand

    def _estimate_restoration_time(
        self, expected_failures: float, crew_demand: float, scenario: ScenarioParameters
    ) -> float:
        """Estimate time to restore service."""
        # Assume 4-person crews working 12-hour shifts
        crews_available = 10.0  # Typical emergency response capacity
        daily_crew_capacity = crews_available * 2.0  # 2 shifts per day

        days_to_restore = crew_demand / daily_crew_capacity
        hours_to_restore = days_to_restore * 24.0

        return hours_to_restore

    def _calculate_risk_distribution(self, adjusted_risks: pd.Series) -> Dict[str, int]:
        """Calculate distribution of poles by risk level."""
        return {
            "low": ((adjusted_risks < 0.3).sum()),
            "medium": ((adjusted_risks >= 0.3) & (adjusted_risks < 0.7)).sum(),
            "high": ((adjusted_risks >= 0.7) & (adjusted_risks < 0.9)).sum(),
            "critical": (adjusted_risks >= 0.9).sum(),
        }
