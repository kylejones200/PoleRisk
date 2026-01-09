"""
Advanced analytics module for soil moisture analysis.

This module provides advanced time series analysis, trend detection,
and climatology features for comprehensive soil moisture data analysis.
"""

from .time_series import TimeSeriesAnalyzer, TrendAnalyzer
from .climatology import ClimatologyAnalyzer, SeasonalAnalyzer

try:
    from .drought import DroughtAnalyzer, DroughtIndex

    DROUGHT_AVAILABLE = True
except ImportError:
    DROUGHT_AVAILABLE = False

try:
    from .statistical_tests import StatisticalTestSuite

    STATISTICAL_TESTS_AVAILABLE = True
except ImportError:
    STATISTICAL_TESTS_AVAILABLE = False

try:
    from .ensemble_forecasting import EnsembleForecaster, ForecastResult

    ENSEMBLE_FORECASTING_AVAILABLE = True
except ImportError:
    ENSEMBLE_FORECASTING_AVAILABLE = False

__all__ = [
    "TimeSeriesAnalyzer",
    "TrendAnalyzer",
    "ClimatologyAnalyzer",
    "SeasonalAnalyzer",
]

if DROUGHT_AVAILABLE:
    __all__.extend(["DroughtAnalyzer", "DroughtIndex"])

if STATISTICAL_TESTS_AVAILABLE:
    __all__.append("StatisticalTestSuite")

if ENSEMBLE_FORECASTING_AVAILABLE:
    __all__.extend(["EnsembleForecaster", "ForecastResult"])
