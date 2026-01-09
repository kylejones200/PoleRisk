# Anomaly Detection Integration

This document describes the integration of the [anomaly-detection-toolkit](https://github.com/kylejones200/anomaly-detection-toolkit) library into the polerisk codebase.

## Overview

The anomaly-detection-toolkit provides comprehensive anomaly detection methods including:

- **Statistical Methods**: Z-score, IQR, seasonal baseline detection
- **Machine Learning Methods**: Isolation Forest, Local Outlier Factor (LOF), Robust Covariance
- **Wavelet Methods**: Wavelet decomposition and denoising for time series
- **Deep Learning Methods**: LSTM and PyTorch autoencoders (optional)
- **Ensemble Methods**: Voting and score combination ensembles

This complements the existing `ts2net` network-based anomaly detection, providing a multi-method approach to identifying anomalies in pole health time series data.

## Installation

The anomaly-detection-toolkit is included as a dependency:

```bash
pip install polerisk
```

For deep learning capabilities (LSTM, PyTorch autoencoders):

```bash
pip install anomaly-detection-toolkit[deep]
```

## Usage

### Basic Single-Method Detection

```python
from polerisk.pole_health.anomaly_detection_integration import (
    AnomalyDetector,
    AnomalyDetectionMethod
)
import numpy as np

# Time series data
moisture_values = np.array([...])  # Your soil moisture time series

# Use Isolation Forest detector
detector = AnomalyDetector(
    method=AnomalyDetectionMethod.ISOLATION_FOREST,
    method_params={"contamination": 0.05}
)

result = detector.detect_time_series_anomalies(
    time_series=moisture_values,
    pole_id="P001",
    time_series_name="soil_moisture"
)

print(f"Anomalies detected: {result.n_anomalies}")
print(f"Anomaly rate: {result.anomaly_rate:.2f}%")
```

### Wavelet-Based Detection (Time Series)

```python
# Wavelet detector is excellent for time series with noise
detector = AnomalyDetector(
    method=AnomalyDetectionMethod.WAVELET,
    method_params={
        "wavelet": "db4",
        "threshold_factor": 2.5,
        "level": 5
    }
)

result = detector.detect_time_series_anomalies(
    time_series=moisture_values,
    timestamps=pd.date_range('2023-01-01', periods=len(moisture_values), freq='D')
)
```

### Ensemble Detection (Recommended)

Ensemble detection combines multiple methods for robust results:

```python
from polerisk.pole_health.anomaly_detection_integration import EnsembleAnomalyDetector

ensemble = EnsembleAnomalyDetector(
    methods=[
        AnomalyDetectionMethod.ISOLATION_FOREST,
        AnomalyDetectionMethod.LOF,
        AnomalyDetectionMethod.WAVELET,
    ],
    method_params={"contamination": 0.05}
)

result = ensemble.detect_ensemble(
    time_series=moisture_values,
    timestamps=timestamps,
    pole_id="P001",
    consensus_threshold=0.5  # 50% of detectors must agree
)

print(f"Consensus anomalies: {result.n_anomalies}")
print(f"Individual results: {result.individual_results}")
print(f"Detector agreement: {result.detector_agreement}")
```

### Integration with Pole Health Assessment

The anomaly detection is automatically integrated into temporal trend analysis:

```python
from polerisk.pole_health.assessment import SoilConditionAnalyzer
from polerisk.pole_health.pole_data import SoilSample

analyzer = SoilConditionAnalyzer()
samples = [...]  # List of SoilSample objects

# Analyze with network analysis enabled
# This includes both ts2net AND anomaly-detection-toolkit
trends = analyzer.analyze_temporal_trends(
    samples=samples,
    use_network_analysis=True  # Enables anomaly detection
)

# Results include anomaly detection metrics
if 'anomaly_detected' in trends:
    print(f"Anomalies: {trends['anomaly_detection_n_anomalies']}")
    print(f"Anomaly rate: {trends['anomaly_detection_rate']:.2f}%")
    
    # Individual detector results
    for key, value in trends.items():
        if key.startswith('anomaly_') and key.endswith('_n_anomalies'):
            detector_name = key.replace('anomaly_', '').replace('_n_anomalies', '')
            print(f"{detector_name}: {value} anomalies")
```

## Available Methods

### Statistical Methods

- **ZScoreDetector**: Z-score based detection (good for normal distributions)
- **IQROutlierDetector**: Interquartile Range based detection (robust to outliers)
- **SeasonalBaselineDetector**: Seasonal pattern-aware detection (excellent for time series)

### Machine Learning Methods

- **IsolationForestDetector**: Isolation Forest (good for high-dimensional data)
- **LOFDetector**: Local Outlier Factor (good for local density variations)
- **RobustCovarianceDetector**: Robust covariance (good for multivariate Gaussian)

### Time Series Methods

- **WaveletDetector**: Wavelet-based detection (excellent for noisy time series)
- **LSTMAutoencoderDetector**: LSTM autoencoder (requires `[deep]` extra)
- **PyTorchAutoencoderDetector**: PyTorch autoencoder (requires `[deep]` extra)

### Ensemble Methods

- **VotingEnsemble**: Votes from multiple detectors
- **EnsembleAnomalyDetector**: Custom ensemble with consensus thresholding

## Performance Considerations

1. **Statistical methods** are fastest but may miss complex patterns
2. **ML methods** (Isolation Forest, LOF) are fast and robust
3. **Wavelet methods** are excellent for time series but slightly slower
4. **Deep learning methods** are slowest but most accurate for complex patterns
5. **Ensemble methods** combine robustness with moderate performance overhead

## When to Use Each Method

- **For real-time monitoring**: Use Isolation Forest or Wavelet (fast, accurate)
- **For batch analysis**: Use Ensemble (most robust, slower)
- **For multivariate data**: Use Isolation Forest or LOF
- **For time series with seasonality**: Use SeasonalBaselineDetector or Wavelet
- **For complex patterns**: Use Ensemble or deep learning methods

## Combining with Network Analysis

The anomaly detection works alongside the existing `ts2net` network-based analysis:

```python
# Both methods provide complementary insights:
# - Network analysis: Pattern complexity, periodicity, chaos
# - Anomaly detection: Statistical outliers, unusual patterns

trends = analyzer.analyze_temporal_trends(
    samples=samples,
    use_network_analysis=True  # Enables both
)

# Results include:
# - Network features: density, complexity, entropy
# - Anomaly results: detection counts, scores, rates
# - Combined risk assessment: trend_risk incorporates both
```

## Example

See `examples/anomaly_detection_example.py` for complete examples demonstrating:

- Single method detection
- Wavelet-based detection
- Ensemble detection
- Integration with pole health assessment

Run the example:

```bash
python examples/anomaly_detection_example.py
```

## References

- [anomaly-detection-toolkit GitHub](https://github.com/kylejones200/anomaly-detection-toolkit)
- [PyPI package](https://pypi.org/project/anomaly-detection-toolkit/)
- [Documentation](https://anomaly-detection-toolkit.readthedocs.io/)

