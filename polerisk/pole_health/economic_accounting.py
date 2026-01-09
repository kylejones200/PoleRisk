"""
Economic lifecycle accounting for utility pole management.

Tracks deferred maintenance risk, avoided outage cost, customer minutes interrupted,
replacement timing, and tradeoffs between inspect, treat, reinforce, and replace.
Essential for capital planning and ROI analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MaintenanceAction(Enum):
    """Types of maintenance actions."""

    INSPECT = "inspect"
    TREAT = "treat"  # Preservative treatment
    REINFORCE = "reinforce"  # Structural reinforcement
    REPLACE = "replace"
    NO_ACTION = "no_action"


@dataclass
class MaintenanceCosts:
    """Cost breakdown for maintenance actions."""

    action: MaintenanceAction
    direct_cost: float  # Direct cost of the action
    labor_cost: float
    material_cost: float
    equipment_cost: float
    indirect_cost: float  # Overhead, admin, etc.
    total_cost: float

    # Opportunity costs
    outage_duration_hours: float = 0.0
    outage_cost: float = 0.0
    customer_impact_cost: float = 0.0

    # Total cost including opportunity
    total_cost_with_opportunity: float = 0.0


@dataclass
class AvoidedCosts:
    """Costs avoided by taking maintenance action."""

    avoided_emergency_repair: float
    avoided_outage_cost: float
    avoided_revenue_loss: float
    avoided_regulatory_fines: float
    avoided_customer_compensation: float
    avoided_liability_cost: float
    total_avoided: float

    # Customer impact avoided
    customer_minutes_interrupted_avoided: float
    customers_affected_avoided: int


@dataclass
class DeferredMaintenanceRisk:
    """Risk associated with deferring maintenance."""

    pole_id: str
    current_risk_score: float
    deferred_1_year_risk: float
    deferred_3_year_risk: float
    deferred_5_year_risk: float

    # Expected costs if deferred
    expected_emergency_cost_1yr: float
    expected_emergency_cost_3yr: float
    expected_emergency_cost_5yr: float

    # Probability of failure
    failure_probability_1yr: float
    failure_probability_3yr: float
    failure_probability_5yr: float

    # Recommended action timing
    recommended_action_date: datetime
    urgency: str  # "immediate", "urgent", "soon", "planned"


@dataclass
class LifecycleCostAnalysis:
    """Complete lifecycle cost analysis for a pole."""

    pole_id: str
    analysis_date: datetime
    current_age_years: float
    expected_remaining_life_years: float

    # Cost scenarios by action
    inspect_cost: MaintenanceCosts
    treat_cost: MaintenanceCosts
    reinforce_cost: MaintenanceCosts
    replace_cost: MaintenanceCosts
    no_action_cost: MaintenanceCosts

    # Avoided costs by action
    inspect_avoided: AvoidedCosts
    treat_avoided: AvoidedCosts
    reinforce_avoided: AvoidedCosts
    replace_avoided: AvoidedCosts
    no_action_avoided: AvoidedCosts

    # Net present value (NPV) for each action
    npv_inspect: float
    npv_treat: float
    npv_reinforce: float
    npv_replace: float
    npv_no_action: float

    # ROI for each action
    roi_inspect: float
    roi_treat: float
    roi_reinforce: float
    roi_replace: float
    roi_no_action: float

    # Recommended action
    recommended_action: MaintenanceAction
    recommended_timing: datetime
    recommendation_confidence: float


class EconomicLifecycleAccountant:
    """Economic lifecycle accounting for pole management."""

    def __init__(self, discount_rate: float = 0.05):
        """Initialize economic lifecycle accountant."""
        self.discount_rate = discount_rate  # Annual discount rate for NPV

        # Default cost assumptions (can be customized)
        self.cost_parameters = {
            "inspect": {
                "direct_cost": 150.0,
                "labor_hours": 2.0,
                "labor_rate": 75.0,
                "material_cost": 0.0,
                "equipment_cost": 25.0,
                "outage_duration": 0.0,
            },
            "treat": {
                "direct_cost": 800.0,
                "labor_hours": 4.0,
                "labor_rate": 75.0,
                "material_cost": 500.0,
                "equipment_cost": 100.0,
                "outage_duration": 2.0,
            },
            "reinforce": {
                "direct_cost": 3000.0,
                "labor_hours": 8.0,
                "labor_rate": 100.0,
                "material_cost": 2000.0,
                "equipment_cost": 500.0,
                "outage_duration": 4.0,
            },
            "replace": {
                "direct_cost": 12000.0,
                "labor_hours": 16.0,
                "labor_rate": 100.0,
                "material_cost": 8000.0,
                "equipment_cost": 2000.0,
                "outage_duration": 8.0,
            },
        }

        # Cost per customer-minute of interruption
        self.customer_interruption_cost_per_minute = 2.0

        # Average customers per pole
        self.avg_customers_per_pole = 50

        # Emergency repair cost multiplier
        self.emergency_cost_multiplier = 3.0

        # Regulatory fine per incident
        self.regulatory_fine_per_incident = 50000.0

    def calculate_maintenance_costs(
        self,
        action: MaintenanceAction,
        pole_data: Dict,
        custom_params: Optional[Dict] = None,
    ) -> MaintenanceCosts:
        """Calculate costs for a maintenance action."""

        if action == MaintenanceAction.NO_ACTION:
            return MaintenanceCosts(
                action=action,
                direct_cost=0.0,
                labor_cost=0.0,
                material_cost=0.0,
                equipment_cost=0.0,
                indirect_cost=0.0,
                total_cost=0.0,
                outage_duration_hours=0.0,
                outage_cost=0.0,
                customer_impact_cost=0.0,
                total_cost_with_opportunity=0.0,
            )

        # Get cost parameters
        params_key = action.value if action.value != "reinforce" else "reinforce"
        params = (custom_params or {}).get(params_key) or self.cost_parameters.get(
            params_key, {}
        )

        # Calculate direct costs
        labor_cost = params.get("labor_hours", 0) * params.get("labor_rate", 0)
        material_cost = params.get("material_cost", 0)
        equipment_cost = params.get("equipment_cost", 0)
        direct_cost = params.get("direct_cost", 0)
        indirect_cost = direct_cost * 0.15  # 15% overhead

        total_cost = (
            labor_cost + material_cost + equipment_cost + direct_cost + indirect_cost
        )

        # Calculate outage costs
        outage_duration = params.get("outage_duration", 0.0)
        customers_affected = pole_data.get(
            "customers_served", self.avg_customers_per_pole
        )

        # Outage cost: customer-minutes * cost per minute
        customer_minutes = outage_duration * 60 * customers_affected
        customer_impact_cost = (
            customer_minutes * self.customer_interruption_cost_per_minute
        )

        # Revenue loss (simplified)
        outage_cost = customer_impact_cost * 1.5  # Include revenue loss

        total_cost_with_opportunity = total_cost + outage_cost

        return MaintenanceCosts(
            action=action,
            direct_cost=direct_cost,
            labor_cost=labor_cost,
            material_cost=material_cost,
            equipment_cost=equipment_cost,
            indirect_cost=indirect_cost,
            total_cost=total_cost,
            outage_duration_hours=outage_duration,
            outage_cost=outage_cost,
            customer_impact_cost=customer_impact_cost,
            total_cost_with_opportunity=total_cost_with_opportunity,
        )

    def calculate_avoided_costs(
        self, action: MaintenanceAction, pole_data: Dict, risk_reduction: float
    ) -> AvoidedCosts:
        """
        Calculate costs avoided by taking maintenance action.

        Args:
            action: Maintenance action taken
            pole_data: Pole data including risk scores, customers served, etc.
            risk_reduction: Reduction in failure probability (0.0 to 1.0)
        """
        current_risk = pole_data.get("risk_score", 0.5)
        customers_served = pole_data.get(
            "customers_served", self.avg_customers_per_pole
        )

        # Base failure probability
        base_failure_prob = current_risk * 0.1  # Convert risk score to probability

        # Reduced failure probability after action
        reduced_failure_prob = base_failure_prob * (1.0 - risk_reduction)

        # Expected emergency repair cost if failure occurs
        base_repair_cost = (
            self.cost_parameters["replace"]["total_cost"]
            * self.emergency_cost_multiplier
        )
        avoided_emergency_repair = (
            base_failure_prob - reduced_failure_prob
        ) * base_repair_cost

        # Expected outage cost
        avg_outage_duration_hours = 24.0  # Average outage duration
        customer_minutes_per_failure = avg_outage_duration_hours * 60 * customers_served
        outage_cost_per_failure = (
            customer_minutes_per_failure
            * self.customer_interruption_cost_per_minute
            * 2.0
        )  # Include revenue loss
        avoided_outage_cost = (
            base_failure_prob - reduced_failure_prob
        ) * outage_cost_per_failure

        # Revenue loss (simplified model)
        avoided_revenue_loss = (
            avoided_outage_cost * 0.3
        )  # 30% of outage cost is revenue loss

        # Regulatory fines
        avoided_regulatory_fines = (
            base_failure_prob - reduced_failure_prob
        ) * self.regulatory_fine_per_incident

        # Customer compensation
        avoided_customer_compensation = avoided_outage_cost * 0.1  # 10% compensation

        # Liability costs (property damage, injury, etc.)
        avg_liability_per_failure = 500000.0  # Average liability cost
        avoided_liability_cost = (
            base_failure_prob - reduced_failure_prob
        ) * avg_liability_per_failure

        total_avoided = (
            avoided_emergency_repair
            + avoided_outage_cost
            + avoided_revenue_loss
            + avoided_regulatory_fines
            + avoided_customer_compensation
            + avoided_liability_cost
        )

        # Customer minutes interrupted avoided
        customer_minutes_interrupted_avoided = (
            base_failure_prob - reduced_failure_prob
        ) * customer_minutes_per_failure

        return AvoidedCosts(
            avoided_emergency_repair=avoided_emergency_repair,
            avoided_outage_cost=avoided_outage_cost,
            avoided_revenue_loss=avoided_revenue_loss,
            avoided_regulatory_fines=avoided_regulatory_fines,
            avoided_customer_compensation=avoided_customer_compensation,
            avoided_liability_cost=avoided_liability_cost,
            total_avoided=total_avoided,
            customer_minutes_interrupted_avoided=customer_minutes_interrupted_avoided,
            customers_affected_avoided=int(
                (base_failure_prob - reduced_failure_prob) * customers_served
            ),
        )

    def calculate_deferred_maintenance_risk(
        self, pole_id: str, current_risk_score: float, risk_growth_rate: float = 0.15
    ) -> DeferredMaintenanceRisk:
        """
        Calculate risk of deferring maintenance.

        Args:
            pole_id: Pole identifier
            current_risk_score: Current risk score (0.0 to 1.0)
            risk_growth_rate: Annual risk growth rate (default 15%)
        """
        # Project risk forward
        deferred_1_year_risk = min(1.0, current_risk_score * (1.0 + risk_growth_rate))
        deferred_3_year_risk = min(
            1.0, current_risk_score * (1.0 + risk_growth_rate) ** 3
        )
        deferred_5_year_risk = min(
            1.0, current_risk_score * (1.0 + risk_growth_rate) ** 5
        )

        # Convert risk scores to failure probabilities
        failure_prob_1yr = deferred_1_year_risk * 0.1
        failure_prob_3yr = deferred_3_year_risk * 0.1
        failure_prob_5yr = deferred_5_year_risk * 0.1

        # Expected emergency costs
        base_repair_cost = (
            self.cost_parameters["replace"]["total_cost"]
            * self.emergency_cost_multiplier
        )
        expected_emergency_cost_1yr = failure_prob_1yr * base_repair_cost
        expected_emergency_cost_3yr = failure_prob_3yr * base_repair_cost
        expected_emergency_cost_5yr = failure_prob_5yr * base_repair_cost

        # Determine recommended action timing
        if current_risk_score > 0.7:
            recommended_action_date = datetime.now()
            urgency = "immediate"
        elif current_risk_score > 0.5:
            recommended_action_date = datetime.now() + timedelta(days=90)
            urgency = "urgent"
        elif deferred_1_year_risk > 0.7:
            recommended_action_date = datetime.now() + timedelta(days=180)
            urgency = "soon"
        else:
            recommended_action_date = datetime.now() + timedelta(days=365)
            urgency = "planned"

        return DeferredMaintenanceRisk(
            pole_id=pole_id,
            current_risk_score=current_risk_score,
            deferred_1_year_risk=deferred_1_year_risk,
            deferred_3_year_risk=deferred_3_year_risk,
            deferred_5_year_risk=deferred_5_year_risk,
            expected_emergency_cost_1yr=expected_emergency_cost_1yr,
            expected_emergency_cost_3yr=expected_emergency_cost_3yr,
            expected_emergency_cost_5yr=expected_emergency_cost_5yr,
            failure_probability_1yr=failure_prob_1yr,
            failure_probability_3yr=failure_prob_3yr,
            failure_probability_5yr=failure_prob_5yr,
            recommended_action_date=recommended_action_date,
            urgency=urgency,
        )

    def analyze_lifecycle_costs(
        self,
        pole_id: str,
        pole_data: Dict,
        current_age_years: float,
        expected_remaining_life: float,
    ) -> LifecycleCostAnalysis:
        """Perform complete lifecycle cost analysis."""

        # Risk reduction factors by action
        risk_reductions = {
            MaintenanceAction.INSPECT: 0.0,  # Inspection doesn't reduce risk, just updates knowledge
            MaintenanceAction.TREAT: 0.3,  # Treatment reduces risk by 30%
            MaintenanceAction.REINFORCE: 0.5,  # Reinforcement reduces risk by 50%
            MaintenanceAction.REPLACE: 1.0,  # Replacement eliminates risk (new pole)
            MaintenanceAction.NO_ACTION: 0.0,
        }

        # Calculate costs for each action
        costs = {}
        avoided = {}
        npvs = {}
        rois = {}

        for action in MaintenanceAction:
            costs[action] = self.calculate_maintenance_costs(action, pole_data)
            risk_reduction = risk_reductions.get(action, 0.0)
            avoided[action] = self.calculate_avoided_costs(
                action, pole_data, risk_reduction
            )

            # Calculate NPV (10-year horizon)
            npv = self._calculate_npv(costs[action], avoided[action], years=10)
            npvs[action] = npv

            # Calculate ROI
            if costs[action].total_cost > 0:
                roi = (
                    avoided[action].total_avoided - costs[action].total_cost
                ) / costs[action].total_cost
            else:
                roi = 0.0 if avoided[action].total_avoided == 0 else float("inf")
            rois[action] = roi

        # Recommend action based on NPV and ROI
        best_action = max(
            MaintenanceAction, key=lambda a: npvs[a] if npvs[a] > 0 else -1e9
        )

        # Confidence based on NPV difference
        sorted_actions = sorted(MaintenanceAction, key=lambda a: npvs[a], reverse=True)
        if len(sorted_actions) >= 2:
            npv_diff = npvs[sorted_actions[0]] - npvs[sorted_actions[1]]
            confidence = min(1.0, max(0.5, npv_diff / 10000.0))  # Normalize
        else:
            confidence = 0.8

        # Recommended timing
        deferred_risk = self.calculate_deferred_maintenance_risk(
            pole_id, pole_data.get("risk_score", 0.5)
        )
        recommended_timing = deferred_risk.recommended_action_date

        return LifecycleCostAnalysis(
            pole_id=pole_id,
            analysis_date=datetime.now(),
            current_age_years=current_age_years,
            expected_remaining_life_years=expected_remaining_life,
            inspect_cost=costs[MaintenanceAction.INSPECT],
            treat_cost=costs[MaintenanceAction.TREAT],
            reinforce_cost=costs[MaintenanceAction.REINFORCE],
            replace_cost=costs[MaintenanceAction.REPLACE],
            no_action_cost=costs[MaintenanceAction.NO_ACTION],
            inspect_avoided=avoided[MaintenanceAction.INSPECT],
            treat_avoided=avoided[MaintenanceAction.TREAT],
            reinforce_avoided=avoided[MaintenanceAction.REINFORCE],
            replace_avoided=avoided[MaintenanceAction.REPLACE],
            no_action_avoided=avoided[MaintenanceAction.NO_ACTION],
            npv_inspect=npvs[MaintenanceAction.INSPECT],
            npv_treat=npvs[MaintenanceAction.TREAT],
            npv_reinforce=npvs[MaintenanceAction.REINFORCE],
            npv_replace=npvs[MaintenanceAction.REPLACE],
            npv_no_action=npvs[MaintenanceAction.NO_ACTION],
            roi_inspect=rois[MaintenanceAction.INSPECT],
            roi_treat=rois[MaintenanceAction.TREAT],
            roi_reinforce=rois[MaintenanceAction.REINFORCE],
            roi_replace=rois[MaintenanceAction.REPLACE],
            roi_no_action=rois[MaintenanceAction.NO_ACTION],
            recommended_action=best_action,
            recommended_timing=recommended_timing,
            recommendation_confidence=confidence,
        )

    def _calculate_npv(
        self, costs: MaintenanceCosts, avoided: AvoidedCosts, years: int = 10
    ) -> float:
        """Calculate Net Present Value over time horizon."""
        # Initial investment (cost in year 0)
        initial_investment = costs.total_cost

        # Annual benefits (avoided costs, annualized)
        annual_benefit = avoided.total_avoided / years  # Simplified: spread over years

        # Calculate NPV
        npv = -initial_investment
        for year in range(1, years + 1):
            discounted_benefit = annual_benefit / ((1 + self.discount_rate) ** year)
            npv += discounted_benefit

        return npv

    def calculate_fleet_roi(
        self, lifecycle_analyses: List[LifecycleCostAnalysis]
    ) -> Dict[str, float]:
        """Calculate fleet-wide ROI metrics."""
        total_cost = sum(
            getattr(analysis, f"{analysis.recommended_action.value}_cost").total_cost
            for analysis in lifecycle_analyses
        )

        total_avoided = sum(
            getattr(
                analysis, f"{analysis.recommended_action.value}_avoided"
            ).total_avoided
            for analysis in lifecycle_analyses
        )

        total_npv = sum(
            getattr(analysis, f"npv_{analysis.recommended_action.value}")
            for analysis in lifecycle_analyses
        )

        total_customer_minutes_avoided = sum(
            getattr(
                analysis, f"{analysis.recommended_action.value}_avoided"
            ).customer_minutes_interrupted_avoided
            for analysis in lifecycle_analyses
        )

        roi = (total_avoided - total_cost) / total_cost if total_cost > 0 else 0.0

        return {
            "total_investment": total_cost,
            "total_avoided_costs": total_avoided,
            "net_savings": total_avoided - total_cost,
            "roi": roi,
            "npv": total_npv,
            "customer_minutes_avoided": total_customer_minutes_avoided,
            "poles_analyzed": len(lifecycle_analyses),
        }
