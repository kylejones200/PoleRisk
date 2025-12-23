"""
Survival analysis and time-to-failure modeling for utility poles.

Provides survival curves by pole class and environment, with remaining useful
life estimates and confidence bands. Moves beyond binary "failure within year"
predictions to align with capital planning cycles.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from scipy import stats
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)


class PoleClass(Enum):
    """Pole classification based on strength and application."""
    H1 = "H1"  # Heavy duty, transmission
    H2 = "H2"  # Heavy duty, distribution
    H3 = "H3"  # Medium duty
    H4 = "H4"  # Light duty
    H5 = "H5"  # Extra light duty


class EnvironmentType(Enum):
    """Environmental classifications."""
    COASTAL = "coastal"  # High humidity, salt exposure
    ARID = "arid"  # Low moisture, high temperature
    TEMPERATE = "temperate"  # Moderate conditions
    COLD = "cold"  # Freeze-thaw cycles
    TROPICAL = "tropical"  # High humidity, high temperature


@dataclass
class SurvivalCurve:
    """Survival curve data structure."""
    pole_class: PoleClass
    environment: EnvironmentType
    ages: np.ndarray  # Age in years
    survival_probabilities: np.ndarray  # S(t) - probability of survival to age t
    hazard_rates: np.ndarray  # h(t) - instantaneous failure rate
    model_type: str  # "weibull", "exponential", "log_normal", etc.
    model_parameters: Dict[str, float]
    confidence_intervals: Optional[np.ndarray] = None  # (lower, upper) for each age


@dataclass
class RemainingUsefulLife:
    """Remaining useful life estimate for a pole."""
    pole_id: str
    current_age: float
    expected_rul: float  # Expected remaining useful life in years
    median_rul: float  # Median remaining useful life
    confidence_band: Tuple[float, float]  # (5th percentile, 95th percentile)
    survival_probability_1yr: float  # Probability of survival 1 year
    survival_probability_5yr: float  # Probability of survival 5 years
    survival_probability_10yr: float  # Probability of survival 10 years
    failure_risk_1yr: float  # Risk of failure within 1 year
    failure_risk_5yr: float  # Risk of failure within 5 years
    model_version: str = "1.0.0"
    assessment_date: datetime = field(default_factory=datetime.now)


class SurvivalAnalysisModel:
    """Model for survival analysis and time-to-failure prediction."""
    
    def __init__(self):
        """Initialize survival analysis model with baseline curves."""
        
        # Baseline survival parameters by pole class and environment
        # Format: (PoleClass, EnvironmentType) -> (shape, scale) for Weibull distribution
        self.baseline_survival_params = self._initialize_baseline_params()
        
        # Risk modifiers based on pole condition
        self.condition_modifiers = {
            'excellent': 1.2,  # 20% longer life
            'good': 1.1,  # 10% longer life
            'fair': 1.0,  # Baseline
            'poor': 0.8,  # 20% shorter life
            'critical': 0.6,  # 40% shorter life
        }
    
    def calculate_survival_curve(self,
                                pole_class: PoleClass,
                                environment: EnvironmentType,
                                condition_rating: str = 'fair',
                                historical_data: Optional[pd.DataFrame] = None) -> SurvivalCurve:
        """
        Calculate survival curve for a pole class and environment.
        
        Args:
            pole_class: Pole strength class
            environment: Environmental classification
            condition_rating: Current condition rating
            historical_data: Optional historical failure data for fitting
            
        Returns:
            SurvivalCurve with probabilities and hazard rates
        """
        # Get baseline parameters
        shape, scale = self.baseline_survival_params[(pole_class, environment)]
        
        # Apply condition modifier
        condition_mod = self.condition_modifiers.get(condition_rating, 1.0)
        scale = scale * condition_mod
        
        # If historical data provided, fit to data
        if historical_data is not None and len(historical_data) > 10:
            shape, scale = self._fit_weibull_to_data(historical_data)
        
        # Generate age range (0 to 2x expected lifetime)
        max_age = scale * 2.5
        ages = np.linspace(0, max_age, 200)
        
        # Calculate survival probabilities using Weibull distribution
        survival_probs = self._weibull_survival(ages, shape, scale)
        
        # Calculate hazard rates
        hazard_rates = self._weibull_hazard(ages, shape, scale)
        
        # Calculate confidence intervals (bootstrap or analytical)
        confidence_intervals = self._calculate_confidence_intervals(
            ages, shape, scale, historical_data
        )
        
        return SurvivalCurve(
            pole_class=pole_class,
            environment=environment,
            ages=ages,
            survival_probabilities=survival_probs,
            hazard_rates=hazard_rates,
            model_type="weibull",
            model_parameters={'shape': shape, 'scale': scale},
            confidence_intervals=confidence_intervals,
        )
    
    def estimate_remaining_useful_life(self,
                                      pole_id: str,
                                      current_age: float,
                                      pole_class: PoleClass,
                                      environment: EnvironmentType,
                                      condition_rating: str = 'fair',
                                      health_score: Optional[float] = None) -> RemainingUsefulLife:
        """
        Estimate remaining useful life for a specific pole.
        
        Args:
            pole_id: Pole identifier
            current_age: Current age in years
            pole_class: Pole strength class
            environment: Environmental classification
            condition_rating: Current condition rating
            health_score: Optional health score (0-100) to refine estimate
            
        Returns:
            RemainingUsefulLife with estimates and confidence bands
        """
        # Get survival curve
        survival_curve = self.calculate_survival_curve(
            pole_class, environment, condition_rating
        )
        
        # Adjust based on health score if provided
        if health_score is not None:
            # Map health score to condition modifier
            if health_score >= 80:
                condition_mod = self.condition_modifiers['excellent']
            elif health_score >= 60:
                condition_mod = self.condition_modifiers['good']
            elif health_score >= 40:
                condition_mod = self.condition_modifiers['fair']
            elif health_score >= 20:
                condition_mod = self.condition_modifiers['poor']
            else:
                condition_mod = self.condition_modifiers['critical']
            
            # Recalculate with adjusted parameters
            shape = survival_curve.model_parameters['shape']
            scale = survival_curve.model_parameters['scale'] * condition_mod
            survival_probs = self._weibull_survival(survival_curve.ages, shape, scale)
        else:
            scale = survival_curve.model_parameters['scale']
            survival_probs = survival_curve.survival_probabilities
        
        # Find survival probability at current age
        if current_age > survival_curve.ages[-1]:
            # Pole is older than model range, extrapolate
            current_survival_prob = 0.01  # Very low
        else:
            current_survival_prob = np.interp(
                current_age, survival_curve.ages, survival_probs
            )
        
        # Conditional survival: S(t|age) = S(t+age) / S(age)
        conditional_survival = survival_probs / current_survival_prob
        
        # Calculate remaining useful life statistics
        # Expected RUL = integral of conditional survival
        future_ages = survival_curve.ages - current_age
        future_ages = future_ages[future_ages > 0]
        conditional_survival_future = conditional_survival[survival_curve.ages > current_age]
        
        if len(conditional_survival_future) > 0:
            # Expected RUL (mean of conditional survival)
            expected_rul = np.trapz(conditional_survival_future, future_ages)
            
            # Median RUL (age where survival = 0.5)
            median_rul = np.interp(0.5, conditional_survival_future[::-1], 
                                  future_ages[::-1])
            
            # Confidence band (5th and 95th percentiles)
            lower_bound = np.interp(0.95, conditional_survival_future[::-1], 
                                   future_ages[::-1])
            upper_bound = np.interp(0.05, conditional_survival_future[::-1], 
                                   future_ages[::-1])
        else:
            expected_rul = 0.0
            median_rul = 0.0
            lower_bound = 0.0
            upper_bound = 0.0
        
        # Calculate survival probabilities at specific time horizons
        horizons = [1, 5, 10]  # years
        survival_probs_horizon = {}
        failure_risks_horizon = {}
        
        for horizon in horizons:
            future_age = current_age + horizon
            if future_age <= survival_curve.ages[-1]:
                survival_prob = np.interp(future_age, survival_curve.ages, survival_probs)
                conditional_prob = survival_prob / current_survival_prob if current_survival_prob > 0 else 0.0
                survival_probs_horizon[f'{horizon}yr'] = conditional_prob
                failure_risks_horizon[f'{horizon}yr'] = 1.0 - conditional_prob
            else:
                survival_probs_horizon[f'{horizon}yr'] = 0.0
                failure_risks_horizon[f'{horizon}yr'] = 1.0
        
        return RemainingUsefulLife(
            pole_id=pole_id,
            current_age=current_age,
            expected_rul=expected_rul,
            median_rul=median_rul,
            confidence_band=(lower_bound, upper_bound),
            survival_probability_1yr=survival_probs_horizon['1yr'],
            survival_probability_5yr=survival_probs_horizon['5yr'],
            survival_probability_10yr=survival_probs_horizon['10yr'],
            failure_risk_1yr=failure_risks_horizon['1yr'],
            failure_risk_5yr=failure_risks_horizon['5yr'],
        )
    
    def batch_estimate_rul(self,
                          poles_df: pd.DataFrame,
                          pole_class_col: str = 'pole_class',
                          environment_col: str = 'environment',
                          age_col: str = 'age_years',
                          condition_col: str = 'condition_rating',
                          health_score_col: Optional[str] = None) -> pd.DataFrame:
        """
        Estimate remaining useful life for a batch of poles.
        
        Args:
            poles_df: DataFrame with pole information
            pole_class_col: Column name for pole class
            environment_col: Column name for environment type
            age_col: Column name for current age
            condition_col: Column name for condition rating
            health_score_col: Optional column name for health score
            
        Returns:
            DataFrame with RUL estimates added
        """
        results = []
        
        for _, row in poles_df.iterrows():
            try:
                pole_class = PoleClass(row[pole_class_col])
                environment = EnvironmentType(row[environment_col])
                age = float(row[age_col])
                condition = row[condition_col] if condition_col in row else 'fair'
                health_score = row[health_score_col] if health_score_col and health_score_col in row else None
                
                rul = self.estimate_remaining_useful_life(
                    pole_id=str(row.get('pole_id', 'unknown')),
                    current_age=age,
                    pole_class=pole_class,
                    environment=environment,
                    condition_rating=condition,
                    health_score=health_score,
                )
                
                results.append({
                    'pole_id': rul.pole_id,
                    'expected_rul': rul.expected_rul,
                    'median_rul': rul.median_rul,
                    'rul_lower_95': rul.confidence_band[0],
                    'rul_upper_95': rul.confidence_band[1],
                    'survival_prob_1yr': rul.survival_probability_1yr,
                    'survival_prob_5yr': rul.survival_probability_5yr,
                    'survival_prob_10yr': rul.survival_probability_10yr,
                    'failure_risk_1yr': rul.failure_risk_1yr,
                    'failure_risk_5yr': rul.failure_risk_5yr,
                })
            except Exception as e:
                logger.warning(f"Error estimating RUL for pole {row.get('pole_id', 'unknown')}: {e}")
                continue
        
        return pd.DataFrame(results)
    
    def _weibull_survival(self, t: np.ndarray, shape: float, scale: float) -> np.ndarray:
        """Calculate Weibull survival function S(t) = exp(-(t/scale)^shape)."""
        return np.exp(-np.power(t / scale, shape))
    
    def _weibull_hazard(self, t: np.ndarray, shape: float, scale: float) -> np.ndarray:
        """Calculate Weibull hazard function h(t) = (shape/scale) * (t/scale)^(shape-1)."""
        return (shape / scale) * np.power(t / scale, shape - 1)
    
    def _fit_weibull_to_data(self, data: pd.DataFrame) -> Tuple[float, float]:
        """Fit Weibull distribution to historical failure data."""
        # Assume data has 'age_at_failure' column
        if 'age_at_failure' not in data.columns:
            raise ValueError("Data must have 'age_at_failure' column")
        
        ages = data['age_at_failure'].values
        ages = ages[ages > 0]  # Remove zeros
        
        if len(ages) < 5:
            # Not enough data, return defaults
            return (2.0, 50.0)
        
        # Fit Weibull using maximum likelihood estimation
        # Using scipy's weibull_min distribution
        params = stats.weibull_min.fit(ages, floc=0)  # floc=0 forces location=0
        shape, scale = params[0], params[2]  # c (shape), scale
        
        return (shape, scale)
    
    def _calculate_confidence_intervals(self,
                                      ages: np.ndarray,
                                      shape: float,
                                      scale: float,
                                      historical_data: Optional[pd.DataFrame]) -> np.ndarray:
        """Calculate confidence intervals for survival probabilities."""
        # Simplified: use analytical approximation
        # In production, use bootstrap or Bayesian methods
        
        # Standard error approximation for Weibull
        n = len(historical_data) if historical_data is not None else 100
        se_factor = 1.96 / np.sqrt(n)  # 95% confidence
        
        survival_probs = self._weibull_survival(ages, shape, scale)
        
        # Approximate bounds
        lower = np.clip(survival_probs - se_factor * 0.1, 0, 1)
        upper = np.clip(survival_probs + se_factor * 0.1, 0, 1)
        
        return np.column_stack([lower, upper])
    
    def _initialize_baseline_params(self) -> Dict[Tuple[PoleClass, EnvironmentType], Tuple[float, float]]:
        """Initialize baseline Weibull parameters (shape, scale) by pole class and environment."""
        params = {}
        
        # H1 poles (strongest, longest life)
        params[(PoleClass.H1, EnvironmentType.TEMPERATE)] = (2.5, 70.0)
        params[(PoleClass.H1, EnvironmentType.COASTAL)] = (2.5, 60.0)
        params[(PoleClass.H1, EnvironmentType.ARID)] = (2.5, 75.0)
        params[(PoleClass.H1, EnvironmentType.COLD)] = (2.5, 65.0)
        params[(PoleClass.H1, EnvironmentType.TROPICAL)] = (2.5, 55.0)
        
        # H2 poles
        params[(PoleClass.H2, EnvironmentType.TEMPERATE)] = (2.3, 60.0)
        params[(PoleClass.H2, EnvironmentType.COASTAL)] = (2.3, 50.0)
        params[(PoleClass.H2, EnvironmentType.ARID)] = (2.3, 65.0)
        params[(PoleClass.H2, EnvironmentType.COLD)] = (2.3, 55.0)
        params[(PoleClass.H2, EnvironmentType.TROPICAL)] = (2.3, 45.0)
        
        # H3 poles
        params[(PoleClass.H3, EnvironmentType.TEMPERATE)] = (2.2, 50.0)
        params[(PoleClass.H3, EnvironmentType.COASTAL)] = (2.2, 42.0)
        params[(PoleClass.H3, EnvironmentType.ARID)] = (2.2, 55.0)
        params[(PoleClass.H3, EnvironmentType.COLD)] = (2.2, 45.0)
        params[(PoleClass.H3, EnvironmentType.TROPICAL)] = (2.2, 38.0)
        
        # H4 poles
        params[(PoleClass.H4, EnvironmentType.TEMPERATE)] = (2.0, 40.0)
        params[(PoleClass.H4, EnvironmentType.COASTAL)] = (2.0, 35.0)
        params[(PoleClass.H4, EnvironmentType.ARID)] = (2.0, 45.0)
        params[(PoleClass.H4, EnvironmentType.COLD)] = (2.0, 38.0)
        params[(PoleClass.H4, EnvironmentType.TROPICAL)] = (2.0, 32.0)
        
        # H5 poles (lightest, shortest life)
        params[(PoleClass.H5, EnvironmentType.TEMPERATE)] = (1.8, 35.0)
        params[(PoleClass.H5, EnvironmentType.COASTAL)] = (1.8, 30.0)
        params[(PoleClass.H5, EnvironmentType.ARID)] = (1.8, 38.0)
        params[(PoleClass.H5, EnvironmentType.COLD)] = (1.8, 32.0)
        params[(PoleClass.H5, EnvironmentType.TROPICAL)] = (1.8, 28.0)
        
        return params

