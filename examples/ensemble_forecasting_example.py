"""
Example demonstrating ensemble forecasting for pole health degradation.

This example shows how to use the EnsembleForecaster which combines
classification (direction) and regression (magnitude) models for improved
forecasting accuracy compared to standard ARIMA methods.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from polerisk.advanced.ensemble_forecasting import EnsembleForecaster
    from polerisk.visualization.time_series_viz import (
        plot_time_series_enhanced,
        plot_decomposition_enhanced,
    )

    IMPORTS_AVAILABLE = True
except ImportError as e:
    logger.error(f"Import error: {e}")
    IMPORTS_AVAILABLE = False


def generate_sample_degradation_data(n_days=365, noise_level=0.02):
    """Generate sample pole health degradation time series."""
    dates = pd.date_range(start="2023-01-01", periods=n_days, freq="D")

    # Simulate degradation: declining health over time with seasonal variations
    trend = np.linspace(100, 70, n_days)  # Health score declining

    # Seasonal pattern (worse in winter/spring due to moisture)
    seasonal = 5 * np.sin(2 * np.pi * np.arange(n_days) / 365.25 - np.pi / 2)

    # Random noise
    noise = np.random.normal(0, noise_level * 100, n_days)

    # Some random "shocks" (maintenance events improving health)
    shocks = np.zeros(n_days)
    shock_indices = np.random.choice(n_days, size=5, replace=False)
    shocks[shock_indices] = np.random.uniform(5, 15, 5)

    health_scores = trend + seasonal + noise + shocks
    health_scores = np.clip(health_scores, 0, 100)  # Clamp to valid range

    return pd.Series(health_scores, index=dates, name="health_score")


def example_ensemble_forecasting():
    """Example using ensemble forecasting."""
    if not IMPORTS_AVAILABLE:
        logger.error("Required imports not available")
        return

    logger.info("\n=== Ensemble Forecasting Example ===")

    # Generate sample data
    data = generate_sample_degradation_data(n_days=500, noise_level=0.03)
    logger.info(f"Generated {len(data)} days of pole health data")

    # Create forecaster
    forecaster = EnsembleForecaster(
        n_lags=3,  # Use 3 lagged values
        test_size=0.2,  # Hold out 20% for testing
        random_state=42,
        n_estimators_classifier=100,
        n_estimators_regressor=100,
    )

    # Fit and predict
    logger.info("Fitting ensemble model...")
    result = forecaster.fit_predict(data, return_metrics=True)

    logger.info("\nForecast Results:")
    logger.info(f"  Classification Accuracy: {result.classification_accuracy:.3f}")
    logger.info(f"  Regression MAE: {result.regression_mae:.3f}")
    logger.info(f"  Ensemble MAE: {result.mae:.3f}")
    logger.info(f"  Ensemble RMSE: {result.rmse:.3f}")
    logger.info(f"  Direction Accuracy: {result.direction_accuracy:.3f}")

    if result.arima_mae is not None:
        logger.info(f"  ARIMA MAE (comparison): {result.arima_mae:.3f}")
        improvement = ((result.arima_mae - result.mae) / result.arima_mae) * 100
        logger.info(f"  Improvement over ARIMA: {improvement:.1f}%")

    # Visualize results
    if result.timestamps is not None and len(result.timestamps) > 0:
        plot_comparison(data, result)

    return result


def plot_comparison(data, result):
    """Plot actual vs predicted values."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot full time series
    axes[0].plot(
        data.index,
        data.values,
        label="Actual Health Score",
        linewidth=2,
        alpha=0.7,
        color="black",
    )

    # Plot predictions
    if result.timestamps is not None and len(result.timestamps) == len(
        result.predictions
    ):
        axes[0].plot(
            result.timestamps,
            result.predictions,
            label="Ensemble Forecast",
            linewidth=2,
            alpha=0.8,
            color="red",
            linestyle="--",
        )

        # Add vertical line to show train/test split
        split_point = result.timestamps[0]
        axes[0].axvline(
            split_point,
            color="gray",
            linestyle=":",
            alpha=0.5,
            label="Train/Test Split",
        )

    axes[0].set_title("Ensemble Forecast vs Actual", fontsize=14)
    axes[0].set_ylabel("Health Score")
    axes[0].legend(frameon=False)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # Plot residuals
    if result.actual is not None and len(result.actual) == len(result.predictions):
        residuals = result.actual - result.predictions
        axes[1].plot(
            result.timestamps,
            residuals,
            linewidth=1.5,
            alpha=0.7,
            color="blue",
        )
        axes[1].axhline(0, color="red", linestyle="--", linewidth=1, alpha=0.5)
        axes[1].fill_between(result.timestamps, residuals, 0, alpha=0.2, color="blue")

    axes[1].set_title(f"Residuals (MAE: {result.mae:.2f})", fontsize=14)
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Residual")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig("ensemble_forecast_comparison.png", dpi=300, bbox_inches="tight")
    logger.info("Saved forecast comparison plot to ensemble_forecast_comparison.png")
    plt.close()


def example_decomposition_visualization():
    """Example using enhanced decomposition visualization."""
    if not IMPORTS_AVAILABLE:
        logger.error("Required imports not available")
        return

    logger.info("\n=== Decomposition Visualization Example ===")

    # Generate sample data
    data = generate_sample_degradation_data(n_days=730, noise_level=0.02)

    # Create decomposition plot
    plot_decomposition_enhanced(
        data,
        model="additive",
        title="Pole Health Score Decomposition",
        filename="pole_health_decomposition.png",
        period=365,  # Annual cycle
    )

    logger.info("Saved decomposition plot to pole_health_decomposition.png")


def example_time_series_plotting():
    """Example using enhanced time series plotting."""
    if not IMPORTS_AVAILABLE:
        logger.error("Required imports not available")
        return

    logger.info("\n=== Enhanced Time Series Plotting Example ===")

    # Generate sample data
    dates = pd.date_range(start="2023-01-01", periods=365, freq="D")
    health_scores = generate_sample_degradation_data(n_days=365)

    # Create DataFrame
    df = pd.DataFrame(
        {
            "date": dates,
            "health_score": health_scores.values,
            "maintenance_cost": np.random.uniform(100, 500, 365),
        }
    )

    # Plot multiple series
    plot_time_series_enhanced(
        df,
        time_column="date",
        value_columns=["health_score", "maintenance_cost"],
        title="Pole Health Metrics Over Time",
        filename="pole_health_time_series.png",
    )

    logger.info("Saved time series plot to pole_health_time_series.png")


def main():
    """Run all examples."""
    logger.info("=" * 60)
    logger.info("Ensemble Forecasting & Enhanced Visualization Examples")
    logger.info("=" * 60)

    try:
        # Example 1: Ensemble forecasting
        result = example_ensemble_forecasting()

        # Example 2: Decomposition visualization
        example_decomposition_visualization()

        # Example 3: Enhanced time series plotting
        example_time_series_plotting()

        logger.info("\n" + "=" * 60)
        logger.info("All examples completed successfully!")
        logger.info("=" * 60)

    except Exception as e:
        logger.exception(f"Error running examples: {e}")


if __name__ == "__main__":
    main()
