"""
Example demonstrating anomaly detection integration with pole health assessment.

This example shows how to use the anomaly-detection-toolkit integration
to detect anomalies in time series data.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import polerisk modules
from polerisk.pole_health.anomaly_detection_integration import (
    AnomalyDetector,
    AnomalyDetectionMethod,
    EnsembleAnomalyDetector,
)
from polerisk.pole_health.assessment import SoilConditionAnalyzer
from polerisk.pole_health.pole_data import SoilSample


# Generate sample soil moisture time series with anomalies
def generate_sample_data(n_samples=365, anomaly_indices=None):
    """Generate sample soil moisture data with injected anomalies."""
    if anomaly_indices is None:
        anomaly_indices = [50, 150, 250, 300]

    # Generate baseline seasonal pattern
    dates = pd.date_range(start="2023-01-01", periods=n_samples, freq="D")

    # Seasonal pattern: higher moisture in winter/spring
    seasonal_pattern = 0.25 + 0.1 * np.sin(2 * np.pi * np.arange(n_samples) / 365.25)

    # Add noise
    noise = np.random.normal(0, 0.02, n_samples)

    # Create baseline
    moisture_values = seasonal_pattern + noise

    # Inject anomalies
    for idx in anomaly_indices:
        if idx < n_samples:
            # Spike anomaly
            moisture_values[idx] += 0.3

    return dates, moisture_values


def example_single_method_detection():
    """Example using a single detection method."""
    logger.info("\n=== Single Method Anomaly Detection ===")

    dates, moisture_values = generate_sample_data()

    # Use Isolation Forest detector
    detector = AnomalyDetector(
        method=AnomalyDetectionMethod.ISOLATION_FOREST,
        method_params={"contamination": 0.05, "n_estimators": 200},
    )

    result = detector.detect_time_series_anomalies(
        time_series=moisture_values,
        timestamps=dates,
        pole_id="P001",
        time_series_name="soil_moisture",
    )

    logger.info(f"Method: {result.method.value}")
    logger.info(
        f"Anomalies detected: {result.n_anomalies} ({result.anomaly_rate:.2f}%)"
    )
    logger.info(f"Mean anomaly score: {result.mean_score:.4f}")
    logger.info(f"Max anomaly score: {result.max_score:.4f}")

    return result


def example_wavelet_detection():
    """Example using wavelet-based detection for time series."""
    logger.info("\n=== Wavelet-Based Anomaly Detection ===")

    dates, moisture_values = generate_sample_data()

    # Use Wavelet detector (good for time series)
    detector = AnomalyDetector(
        method=AnomalyDetectionMethod.WAVELET,
        method_params={"wavelet": "db4", "threshold_factor": 2.5, "level": 5},
    )

    result = detector.detect_time_series_anomalies(
        time_series=moisture_values,
        timestamps=dates,
        pole_id="P002",
        time_series_name="soil_moisture",
    )

    logger.info(f"Method: {result.method.value}")
    logger.info(
        f"Anomalies detected: {result.n_anomalies} ({result.anomaly_rate:.2f}%)"
    )

    return result


def example_ensemble_detection():
    """Example using ensemble detection for robust results."""
    logger.info("\n=== Ensemble Anomaly Detection ===")

    dates, moisture_values = generate_sample_data()

    # Use ensemble detector (combines multiple methods)
    ensemble_detector = EnsembleAnomalyDetector(
        methods=[
            AnomalyDetectionMethod.ISOLATION_FOREST,
            AnomalyDetectionMethod.LOF,
            AnomalyDetectionMethod.WAVELET,
        ],
        method_params={
            "contamination": 0.05,
        },
    )

    result = ensemble_detector.detect_ensemble(
        time_series=moisture_values,
        timestamps=dates,
        pole_id="P003",
        time_series_name="soil_moisture",
        consensus_threshold=0.5,  # 50% of detectors must agree
    )

    logger.info(
        f"Ensemble consensus anomalies: {result.n_anomalies} ({result.anomaly_rate:.2f}%)"
    )

    # Show individual detector results
    logger.info("\nIndividual detector results:")
    for method_name, individual_result in result.individual_results.items():
        logger.info(
            f"  {method_name}: {individual_result.n_anomalies} anomalies ({individual_result.anomaly_rate:.2f}%)"
        )

    # Show detector agreement
    logger.info("\nDetector agreement:")
    for pair, agreement in result.detector_agreement.items():
        logger.info(f"  {pair}: {agreement:.1f}% agreement")

    return result


def example_integration_with_assessment():
    """Example integrating anomaly detection with pole health assessment."""
    logger.info("\n=== Integration with Pole Health Assessment ===")

    # Create sample soil samples with anomalies
    dates, moisture_values = generate_sample_data(n_samples=90)  # Quarterly for 1 year

    samples = []
    for i, (date, moisture) in enumerate(zip(dates, moisture_values)):
        sample = SoilSample(
            pole_id="P004",
            sample_date=date,
            depth_inches=12.0,
            moisture_content=moisture,
            ph=7.0,
            bulk_density=1.5,
            electrical_conductivity=1.0,
            bearing_capacity=200.0,
            soil_type="sandy_loam",
            data_quality="good",
        )
        samples.append(sample)

    # Use SoilConditionAnalyzer with network analysis (which includes anomaly detection)
    analyzer = SoilConditionAnalyzer()

    # Analyze temporal trends with network analysis enabled
    # This now includes both ts2net network analysis AND anomaly-detection-toolkit
    trends = analyzer.analyze_temporal_trends(
        samples=samples,
        use_network_analysis=True,  # Enables both network and anomaly detection
    )

    logger.info("\nTemporal trend analysis results:")
    logger.info(f"Trend risk: {trends.get('trend_risk', 0):.3f}")
    logger.info(f"Moisture trend: {trends.get('moisture_trend', 0):.6f}")
    logger.info(f"Trend direction: {trends.get('trend_direction', 'unknown')}")

    # Show anomaly detection results if available
    if "anomaly_detected" in trends:
        logger.info(f"\nAnomaly detection results:")
        logger.info(
            f"  Anomalies detected: {trends.get('anomaly_detection_n_anomalies', 0)}"
        )
        logger.info(f"  Anomaly rate: {trends.get('anomaly_detection_rate', 0):.2f}%")
        logger.info(
            f"  Mean anomaly score: {trends.get('anomaly_detection_mean_score', 0):.4f}"
        )

        # Show individual detector results
        for key, value in trends.items():
            if key.startswith("anomaly_") and key.endswith("_n_anomalies"):
                detector_name = key.replace("anomaly_", "").replace("_n_anomalies", "")
                logger.info(f"  {detector_name}: {value} anomalies")

    return trends


def main():
    """Run all examples."""
    logger.info("=" * 60)
    logger.info("Anomaly Detection Integration Examples")
    logger.info("=" * 60)

    try:
        # Example 1: Single method
        result1 = example_single_method_detection()

        # Example 2: Wavelet method
        result2 = example_wavelet_detection()

        # Example 3: Ensemble detection
        result3 = example_ensemble_detection()

        # Example 4: Integration with assessment
        trends = example_integration_with_assessment()

        logger.info("\n" + "=" * 60)
        logger.info("All examples completed successfully!")
        logger.info("=" * 60)

    except ImportError as e:
        logger.error(f"Error: {e}")
        logger.error("Make sure anomaly-detection-toolkit is installed:")
        logger.error("  pip install anomaly-detection-toolkit")
    except Exception as e:
        logger.exception(f"Error running examples: {e}")


if __name__ == "__main__":
    main()
