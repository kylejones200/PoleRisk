"""
Validation at fleet scale for utility pole assessments.

Includes backtests by region and year, reports alert rates, missed failures,
and cost impact. Provides evidence that executives need, not just promises.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationMetrics:
    """Validation metrics for model performance."""

    region: str
    year: int

    # Prediction accuracy
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int

    # Calculated metrics
    sensitivity: float  # Recall: TP / (TP + FN)
    specificity: float  # TN / (TN + FP)
    precision: float  # TP / (TP + FP)
    accuracy: float  # (TP + TN) / (TP + TN + FP + FN)
    f1_score: float

    # Alert metrics
    total_alerts: int
    alert_rate: float  # Alerts per 1000 poles
    false_alert_rate: float  # False positives per 1000 poles

    # Failure metrics
    total_failures: int
    missed_failures: int
    detected_failures: int
    failure_rate: float  # Failures per 1000 poles

    # Cost impact
    total_cost_impact: float
    avoided_costs: float
    false_alert_costs: float
    missed_failure_costs: float
    net_savings: float

    # Confidence intervals
    sensitivity_ci: Tuple[float, float] = (0.0, 0.0)
    precision_ci: Tuple[float, float] = (0.0, 0.0)


@dataclass
class BacktestResult:
    """Results from a backtest analysis."""

    backtest_id: str
    backtest_date: datetime
    region: str
    year: int
    start_date: datetime
    end_date: datetime

    # Data summary
    total_poles: int
    poles_with_assessments: int
    poles_with_failures: int

    # Validation metrics
    metrics: ValidationMetrics

    # Performance by risk category
    performance_by_risk: Dict[str, Dict[str, float]]  # risk_level -> metrics

    # Performance over time
    monthly_performance: List[Dict[str, Any]] = field(default_factory=list)

    # Recommendations
    recommendations: List[str] = field(default_factory=list)


@dataclass
class FleetValidationReport:
    """Comprehensive fleet-wide validation report."""

    report_id: str
    report_date: datetime
    reporting_period: Tuple[datetime, datetime]

    # Overall metrics
    overall_metrics: ValidationMetrics

    # Backtests by region and year
    backtests: List[BacktestResult]

    # Summary by region
    metrics_by_region: Dict[str, ValidationMetrics]

    # Summary by year
    metrics_by_year: Dict[int, ValidationMetrics]

    # Trend analysis
    trend_analysis: Dict[str, List[float]]  # metric_name -> values over time

    # Key findings
    key_findings: List[str]

    # Action items
    action_items: List[str] = field(default_factory=list)


class FleetValidationAnalyzer:
    """Analyze and validate model performance at fleet scale."""

    def __init__(
        self,
        failure_cost_per_incident: float = 100000.0,
        false_alert_cost: float = 5000.0,
    ):
        """Initialize fleet validation analyzer."""
        self.failure_cost_per_incident = failure_cost_per_incident
        self.false_alert_cost = false_alert_cost

    def run_backtest(
        self,
        predictions_df: pd.DataFrame,
        actual_failures_df: pd.DataFrame,
        region: str,
        year: int,
        risk_threshold: float = 0.7,
    ) -> BacktestResult:
        """
        Run a backtest comparing predictions to actual failures.

        Args:
            predictions_df: DataFrame with columns ['pole_id', 'prediction_date', 'risk_score', 'predicted_failure_probability']
            actual_failures_df: DataFrame with columns ['pole_id', 'failure_date']
            region: Region name
            year: Year of backtest
            risk_threshold: Risk score threshold for alert (default 0.7)
        """

        backtest_id = f"BACKTEST_{region}_{year}_{datetime.now().strftime('%Y%m%d')}"
        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31)

        # Merge predictions and actuals
        merged = predictions_df.merge(
            actual_failures_df, on="pole_id", how="outer", suffixes=("_pred", "_actual")
        )

        # Define positive predictions (alerts)
        merged["predicted_failure"] = (
            merged["predicted_failure_probability"] >= risk_threshold
        )
        merged["actual_failure"] = merged["failure_date"].notna()

        # Calculate confusion matrix using vectorized operations
        predicted = merged["predicted_failure"].fillna(False)
        actual = merged["actual_failure"].fillna(False)

        tp = (predicted & actual).sum()
        fp = (predicted & ~actual).sum()
        tn = (~predicted & ~actual).sum()
        fn = (~predicted & actual).sum()

        # Calculate metrics
        total = tp + fp + tn + fn
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        accuracy = (tp + tn) / total if total > 0 else 0.0
        f1_score = (
            2 * (precision * sensitivity) / (precision + sensitivity)
            if (precision + sensitivity) > 0
            else 0.0
        )

        # Calculate confidence intervals (Wilson score interval)
        sensitivity_ci = self._wilson_confidence_interval(tp, tp + fn)
        precision_ci = self._wilson_confidence_interval(tp, tp + fp)

        # Constants for rate calculations
        RATE_SCALING_PER_1000 = 1000.0
        COST_AVOIDANCE_FACTOR = 0.5  # Assume 50% cost avoidance from early intervention

        # Alert metrics
        total_alerts = tp + fp
        total_poles = len(predictions_df["pole_id"].unique())
        alert_rate = (
            (total_alerts / total_poles * RATE_SCALING_PER_1000)
            if total_poles > 0
            else 0.0
        )
        false_alert_rate = (
            (fp / total_poles * RATE_SCALING_PER_1000) if total_poles > 0 else 0.0
        )

        # Failure metrics
        total_failures = tp + fn
        missed_failures = fn
        detected_failures = tp
        failure_rate = (
            (total_failures / total_poles * RATE_SCALING_PER_1000)
            if total_poles > 0
            else 0.0
        )

        # Cost impact
        avoided_costs = tp * self.failure_cost_per_incident * COST_AVOIDANCE_FACTOR
        false_alert_costs = fp * self.false_alert_cost
        missed_failure_costs = fn * self.failure_cost_per_incident
        total_cost_impact = avoided_costs - false_alert_costs - missed_failure_costs
        net_savings = avoided_costs - false_alert_costs

        metrics = ValidationMetrics(
            region=region,
            year=year,
            true_positives=int(tp),
            false_positives=int(fp),
            true_negatives=int(tn),
            false_negatives=int(fn),
            sensitivity=sensitivity,
            specificity=specificity,
            precision=precision,
            accuracy=accuracy,
            f1_score=f1_score,
            total_alerts=int(total_alerts),
            alert_rate=alert_rate,
            false_alert_rate=false_alert_rate,
            total_failures=int(total_failures),
            missed_failures=int(missed_failures),
            detected_failures=int(detected_failures),
            failure_rate=failure_rate,
            total_cost_impact=total_cost_impact,
            avoided_costs=avoided_costs,
            false_alert_costs=false_alert_costs,
            missed_failure_costs=missed_failure_costs,
            net_savings=net_savings,
            sensitivity_ci=sensitivity_ci,
            precision_ci=precision_ci,
        )

        # Performance by risk category
        performance_by_risk = self._calculate_performance_by_risk(
            merged, risk_threshold
        )

        # Monthly performance
        monthly_performance = self._calculate_monthly_performance(merged, year)

        # Generate recommendations
        recommendations = self._generate_recommendations(metrics)

        return BacktestResult(
            backtest_id=backtest_id,
            backtest_date=datetime.now(),
            region=region,
            year=year,
            start_date=start_date,
            end_date=end_date,
            total_poles=total_poles,
            poles_with_assessments=len(predictions_df),
            poles_with_failures=len(actual_failures_df),
            metrics=metrics,
            performance_by_risk=performance_by_risk,
            monthly_performance=monthly_performance,
            recommendations=recommendations,
        )

    def create_fleet_validation_report(
        self,
        backtests: List[BacktestResult],
        reporting_period: Tuple[datetime, datetime],
    ) -> FleetValidationReport:
        """Create comprehensive fleet-wide validation report."""

        # Aggregate overall metrics
        total_tp = sum(bt.metrics.true_positives for bt in backtests)
        total_fp = sum(bt.metrics.false_positives for bt in backtests)
        total_tn = sum(bt.metrics.true_negatives for bt in backtests)
        total_fn = sum(bt.metrics.false_negatives for bt in backtests)

        overall_sensitivity = (
            total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        )
        overall_specificity = (
            total_tn / (total_tn + total_fp) if (total_tn + total_fp) > 0 else 0.0
        )
        overall_precision = (
            total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        )
        overall_accuracy = (
            (total_tp + total_tn) / (total_tp + total_fp + total_tn + total_fn)
            if (total_tp + total_fp + total_tn + total_fn) > 0
            else 0.0
        )
        overall_f1 = (
            2
            * (overall_precision * overall_sensitivity)
            / (overall_precision + overall_sensitivity)
            if (overall_precision + overall_sensitivity) > 0
            else 0.0
        )

        total_poles = sum(bt.total_poles for bt in backtests)
        total_alerts = sum(bt.metrics.total_alerts for bt in backtests)
        total_failures = sum(bt.metrics.total_failures for bt in backtests)

        overall_metrics = ValidationMetrics(
            region="ALL",
            year=0,
            true_positives=total_tp,
            false_positives=total_fp,
            true_negatives=total_tn,
            false_negatives=total_fn,
            sensitivity=overall_sensitivity,
            specificity=overall_specificity,
            precision=overall_precision,
            accuracy=overall_accuracy,
            f1_score=overall_f1,
            total_alerts=total_alerts,
            alert_rate=(total_alerts / total_poles * 1000) if total_poles > 0 else 0.0,
            false_alert_rate=(
                (total_fp / total_poles * 1000) if total_poles > 0 else 0.0
            ),
            total_failures=total_failures,
            missed_failures=sum(bt.metrics.missed_failures for bt in backtests),
            detected_failures=total_tp,
            failure_rate=(
                (total_failures / total_poles * 1000) if total_poles > 0 else 0.0
            ),
            total_cost_impact=sum(bt.metrics.total_cost_impact for bt in backtests),
            avoided_costs=sum(bt.metrics.avoided_costs for bt in backtests),
            false_alert_costs=sum(bt.metrics.false_alert_costs for bt in backtests),
            missed_failure_costs=sum(
                bt.metrics.missed_failure_costs for bt in backtests
            ),
            net_savings=sum(bt.metrics.net_savings for bt in backtests),
        )

        # Metrics by region
        metrics_by_region = {}
        for bt in backtests:
            if bt.region not in metrics_by_region:
                metrics_by_region[bt.region] = bt.metrics
            else:
                # Aggregate if multiple backtests for same region
                # Simplified - in production would properly aggregate
                metrics_by_region[bt.region] = bt.metrics

        # Metrics by year
        metrics_by_year = {}
        for bt in backtests:
            metrics_by_year[bt.year] = bt.metrics

        # Trend analysis
        trend_analysis = self._calculate_trends(backtests)

        # Key findings
        key_findings = self._generate_key_findings(overall_metrics, backtests)

        # Action items
        action_items = self._generate_action_items(overall_metrics, backtests)

        return FleetValidationReport(
            report_id=f"VALIDATION_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            report_date=datetime.now(),
            reporting_period=reporting_period,
            overall_metrics=overall_metrics,
            backtests=backtests,
            metrics_by_region=metrics_by_region,
            metrics_by_year=metrics_by_year,
            trend_analysis=trend_analysis,
            key_findings=key_findings,
            action_items=action_items,
        )

    def _wilson_confidence_interval(
        self, successes: float, trials: float, confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate Wilson score confidence interval."""
        if trials == 0:
            return (0.0, 0.0)

        z = 1.96  # For 95% confidence
        p = successes / trials
        denominator = 1 + (z**2 / trials)
        centre_adjusted_probability = (p + (z**2 / (2 * trials))) / denominator
        adjusted_standard_deviation = (
            np.sqrt((p * (1 - p) + z**2 / (4 * trials)) / trials) / denominator
        )

        lower_bound = centre_adjusted_probability - z * adjusted_standard_deviation
        upper_bound = centre_adjusted_probability + z * adjusted_standard_deviation

        return (max(0.0, lower_bound), min(1.0, upper_bound))

    def _calculate_performance_by_risk(
        self, merged_df: pd.DataFrame, risk_threshold: float
    ) -> Dict[str, Dict[str, float]]:
        """Calculate performance metrics by risk category."""
        # Define risk categories
        merged_df["risk_category"] = pd.cut(
            merged_df["predicted_failure_probability"],
            bins=[0, 0.3, 0.5, 0.7, 1.0],
            labels=["low", "medium", "high", "critical"],
        )

        performance = {}
        for category in ["low", "medium", "high", "critical"]:
            cat_data = merged_df[merged_df["risk_category"] == category]
            if len(cat_data) == 0:
                continue

            # Vectorized confusion matrix calculation
            predicted = cat_data["predicted_failure"].fillna(False)
            actual = cat_data["actual_failure"].fillna(False)

            tp = (predicted & actual).sum()
            fp = (predicted & ~actual).sum()
            fn = (~predicted & actual).sum()
            tn = (~predicted & ~actual).sum()

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

            performance[category] = {
                "sensitivity": sensitivity,
                "precision": precision,
                "failure_rate": (tp + fn) / len(cat_data) if len(cat_data) > 0 else 0.0,
            }

        return performance

    def _calculate_monthly_performance(
        self, merged_df: pd.DataFrame, year: int
    ) -> List[Dict[str, Any]]:
        """Calculate performance metrics by month."""
        monthly_perf = []

        for month in range(1, 13):
            month_start = datetime(year, month, 1)
            if month == 12:
                month_end = datetime(year + 1, 1, 1)
            else:
                month_end = datetime(year, month + 1, 1)

            month_data = merged_df[
                (merged_df["prediction_date"] >= month_start)
                & (merged_df["prediction_date"] < month_end)
            ]

            if len(month_data) == 0:
                continue

            # Vectorized confusion matrix calculation
            predicted = month_data["predicted_failure"].fillna(False)
            actual = month_data["actual_failure"].fillna(False)

            tp = (predicted & actual).sum()
            fp = (predicted & ~actual).sum()
            fn = (~predicted & actual).sum()

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

            monthly_perf.append(
                {
                    "month": month,
                    "year": year,
                    "total_assessments": len(month_data),
                    "sensitivity": sensitivity,
                    "precision": precision,
                    "total_failures": tp + fn,
                    "detected_failures": tp,
                }
            )

        return monthly_perf

    def _generate_recommendations(self, metrics: ValidationMetrics) -> List[str]:
        """Generate recommendations based on validation metrics."""
        recommendations = []

        if metrics.sensitivity < 0.85:
            recommendations.append(
                "Sensitivity below target (85%). Consider lowering risk threshold or improving model features."
            )

        if metrics.precision < 0.80:
            recommendations.append(
                "Precision below target (80%). Consider raising risk threshold to reduce false positives."
            )

        if metrics.false_alert_rate > 50:  # More than 50 per 1000 poles
            recommendations.append(
                "False alert rate high. Review threshold and model calibration."
            )

        if metrics.missed_failures > 0:
            recommendations.append(
                f"{metrics.missed_failures} failures missed. Review model performance and adjust accordingly."
            )

        if metrics.net_savings < 0:
            recommendations.append(
                "Net savings negative. Review cost assumptions and model performance."
            )

        return recommendations

    def _calculate_trends(
        self, backtests: List[BacktestResult]
    ) -> Dict[str, List[float]]:
        """Calculate trends over time."""
        # Sort by year
        sorted_backtests = sorted(backtests, key=lambda x: x.year)

        trends = {
            "sensitivity": [bt.metrics.sensitivity for bt in sorted_backtests],
            "precision": [bt.metrics.precision for bt in sorted_backtests],
            "accuracy": [bt.metrics.accuracy for bt in sorted_backtests],
            "f1_score": [bt.metrics.f1_score for bt in sorted_backtests],
            "alert_rate": [bt.metrics.alert_rate for bt in sorted_backtests],
            "net_savings": [bt.metrics.net_savings for bt in sorted_backtests],
        }

        return trends

    def _generate_key_findings(
        self, overall_metrics: ValidationMetrics, backtests: List[BacktestResult]
    ) -> List[str]:
        """Generate key findings from validation."""
        findings = []

        findings.append(f"Overall model sensitivity: {overall_metrics.sensitivity:.1%}")
        findings.append(f"Overall model precision: {overall_metrics.precision:.1%}")
        findings.append(
            f"Total failures detected: {overall_metrics.detected_failures} out of {overall_metrics.total_failures}"
        )
        findings.append(f"Missed failures: {overall_metrics.missed_failures}")
        findings.append(
            f"Net savings from predictions: ${overall_metrics.net_savings:,.0f}"
        )

        if overall_metrics.false_alert_rate > 50:
            findings.append(
                f"High false alert rate: {overall_metrics.false_alert_rate:.1f} per 1000 poles"
            )

        return findings

    def _generate_action_items(
        self, overall_metrics: ValidationMetrics, backtests: List[BacktestResult]
    ) -> List[str]:
        """Generate action items from validation."""
        items = []

        if overall_metrics.sensitivity < 0.90:
            items.append("Improve model sensitivity to reduce missed failures")

        if overall_metrics.precision < 0.85:
            items.append("Reduce false positive rate to improve precision")

        if overall_metrics.missed_failures > 0:
            items.append(
                f"Investigate {overall_metrics.missed_failures} missed failures for model improvement opportunities"
            )

        return items
