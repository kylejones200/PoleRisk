"""
Integration of anomaly-detection-toolkit for comprehensive anomaly detection.

Leverages multiple detection methods from anomaly-detection-toolkit including:
- Statistical methods (Z-score, IQR, seasonal baseline)
- Machine learning methods (Isolation Forest, LOF, Robust Covariance)
- Wavelet methods for time series
- Deep learning methods (LSTM, PyTorch autoencoders)
- Ensemble methods for robust detection

Reference: https://github.com/kylejones200/anomaly-detection-toolkit
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)

# Try importing anomaly-detection-toolkit
try:
    from anomaly_detection_toolkit import (
        ZScoreDetector,
        IQROutlierDetector,
        SeasonalBaselineDetector,
        IsolationForestDetector,
        LOFDetector,
        RobustCovarianceDetector,
        WaveletDetector,
        VotingEnsemble,
    )

    ANOMALY_DETECTION_AVAILABLE = True
except ImportError:
    ANOMALY_DETECTION_AVAILABLE = False
    logger.warning(
        "anomaly-detection-toolkit not available. "
        "Install with: pip install anomaly-detection-toolkit"
    )

# Try importing deep learning detectors (optional)
try:
    from anomaly_detection_toolkit import (
        LSTMAutoencoderDetector,
        PyTorchAutoencoderDetector,
    )

    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False
    logger.debug(
        "Deep learning detectors not available. "
        "Install with: pip install anomaly-detection-toolkit[deep]"
    )


class AnomalyDetectionMethod(Enum):
    """Available anomaly detection methods."""

    ZSCORE = "zscore"
    IQR = "iqr"
    SEASONAL_BASELINE = "seasonal_baseline"
    ISOLATION_FOREST = "isolation_forest"
    LOF = "lof"  # Local Outlier Factor
    ROBUST_COVARIANCE = "robust_covariance"
    WAVELET = "wavelet"
    LSTM = "lstm"
    PYTORCH_AUTOENCODER = "pytorch_autoencoder"
    ENSEMBLE = "ensemble"


@dataclass
class AnomalyDetectionResult:
    """Results from anomaly detection analysis."""

    pole_id: str
    time_series_name: str
    method: AnomalyDetectionMethod

    # Detection results
    predictions: np.ndarray  # -1 for anomalies, 1 for normal
    scores: np.ndarray  # Anomaly scores (higher = more anomalous)

    # Statistics
    n_anomalies: int
    anomaly_rate: float  # Percentage of data points flagged as anomalies
    mean_score: float
    max_score: float
    min_score: float

    # Timestamps (if available)
    timestamps: Optional[pd.DatetimeIndex] = None

    # Metadata
    detection_date: datetime = field(default_factory=datetime.now)
    parameters: Dict = field(default_factory=dict)


@dataclass
class EnsembleAnomalyResult:
    """Results from ensemble anomaly detection."""

    pole_id: str
    time_series_name: str

    # Individual detector results
    individual_results: Dict[str, AnomalyDetectionResult]

    # Ensemble consensus
    consensus_predictions: np.ndarray
    consensus_scores: np.ndarray
    n_anomalies: int
    anomaly_rate: float

    # Agreement metrics
    detector_agreement: Dict[str, float]  # Agreement percentage between detectors

    # Metadata
    detection_date: datetime = field(default_factory=datetime.now)


class AnomalyDetector:
    """Comprehensive anomaly detector using anomaly-detection-toolkit."""

    def __init__(
        self,
        method: AnomalyDetectionMethod = AnomalyDetectionMethod.ENSEMBLE,
        method_params: Optional[Dict] = None,
    ):
        """
        Initialize anomaly detector.

        Args:
            method: Detection method to use
            method_params: Parameters specific to the chosen method
        """
        if not ANOMALY_DETECTION_AVAILABLE:
            raise ImportError(
                "anomaly-detection-toolkit is required. "
                "Install with: pip install anomaly-detection-toolkit"
            )

        self.method = method
        self.method_params = method_params or {}
        self.detector = None
        self._initialize_detector()

    def _initialize_detector(self):
        """Initialize the appropriate detector based on method."""
        if self.method == AnomalyDetectionMethod.ZSCORE:
            self.detector = ZScoreDetector(n_std=self.method_params.get("n_std", 3.0))

        elif self.method == AnomalyDetectionMethod.IQR:
            self.detector = IQROutlierDetector(
                factor=self.method_params.get("factor", 1.5)
            )

        elif self.method == AnomalyDetectionMethod.SEASONAL_BASELINE:
            self.detector = SeasonalBaselineDetector(
                seasonality=self.method_params.get("seasonality", "week"),
                threshold_sigma=self.method_params.get("threshold_sigma", 2.5),
            )

        elif self.method == AnomalyDetectionMethod.ISOLATION_FOREST:
            self.detector = IsolationForestDetector(
                contamination=self.method_params.get("contamination", 0.05),
                n_estimators=self.method_params.get("n_estimators", 200),
                random_state=self.method_params.get("random_state", 42),
            )

        elif self.method == AnomalyDetectionMethod.LOF:
            self.detector = LOFDetector(
                contamination=self.method_params.get("contamination", 0.05),
                n_neighbors=self.method_params.get("n_neighbors", 20),
            )

        elif self.method == AnomalyDetectionMethod.ROBUST_COVARIANCE:
            self.detector = RobustCovarianceDetector(
                contamination=self.method_params.get("contamination", 0.05),
                random_state=self.method_params.get("random_state", 42),
            )

        elif self.method == AnomalyDetectionMethod.WAVELET:
            self.detector = WaveletDetector(
                wavelet=self.method_params.get("wavelet", "db4"),
                threshold_factor=self.method_params.get("threshold_factor", 2.5),
                level=self.method_params.get("level", 5),
            )

        elif self.method == AnomalyDetectionMethod.LSTM:
            if not DEEP_LEARNING_AVAILABLE:
                raise ImportError(
                    "Deep learning detectors require: pip install anomaly-detection-toolkit[deep]"
                )
            self.detector = LSTMAutoencoderDetector(
                window_size=self.method_params.get("window_size", 20),
                lstm_units=self.method_params.get("lstm_units", [32, 16]),
                epochs=self.method_params.get("epochs", 50),
                threshold_std=self.method_params.get("threshold_std", 3.0),
            )

        elif self.method == AnomalyDetectionMethod.PYTORCH_AUTOENCODER:
            if not DEEP_LEARNING_AVAILABLE:
                raise ImportError(
                    "Deep learning detectors require: pip install anomaly-detection-toolkit[deep]"
                )
            self.detector = PyTorchAutoencoderDetector(
                window_size=self.method_params.get("window_size", 24),
                hidden_dims=self.method_params.get("hidden_dims", [64, 16, 4]),
                epochs=self.method_params.get("epochs", 200),
                threshold_std=self.method_params.get("threshold_std", 3.0),
            )

        elif self.method == AnomalyDetectionMethod.ENSEMBLE:
            # Create ensemble of multiple detectors
            detectors = [
                IsolationForestDetector(
                    contamination=self.method_params.get("contamination", 0.05)
                ),
                LOFDetector(
                    contamination=self.method_params.get("contamination", 0.05)
                ),
                RobustCovarianceDetector(
                    contamination=self.method_params.get("contamination", 0.05)
                ),
            ]
            voting_threshold = self.method_params.get("voting_threshold", 2)
            self.detector = VotingEnsemble(detectors, voting_threshold=voting_threshold)

        else:
            raise ValueError(f"Unknown detection method: {self.method}")

    def detect_anomalies(
        self,
        data: Union[np.ndarray, pd.Series, pd.DataFrame],
        pole_id: str = "",
        time_series_name: str = "time_series",
        timestamps: Optional[pd.DatetimeIndex] = None,
    ) -> AnomalyDetectionResult:
        """
        Detect anomalies in time series or multivariate data.

        Args:
            data: Input data (1D array for time series, 2D for multivariate)
            pole_id: Identifier for the pole being analyzed
            time_series_name: Name of the time series
            timestamps: Optional timestamps for the data points

        Returns:
            AnomalyDetectionResult with detection results and statistics
        """
        # Convert to numpy array
        if isinstance(data, pd.Series):
            data_array = data.values
        elif isinstance(data, pd.DataFrame):
            data_array = data.values
        else:
            data_array = np.asarray(data)

        # Handle 1D vs 2D data
        if data_array.ndim == 1:
            # Reshape for scikit-learn compatibility
            data_reshaped = data_array.reshape(-1, 1)
        else:
            data_reshaped = data_array

        # Fit and predict
        try:
            self.detector.fit(data_reshaped)
            predictions, scores = self.detector.fit_predict(data_reshaped)
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            # Return all normal predictions on error
            predictions = np.ones(len(data_reshaped))
            scores = np.zeros(len(data_reshaped))

        # Calculate statistics
        n_anomalies = int((predictions == -1).sum())
        anomaly_rate = (
            (n_anomalies / len(predictions) * 100) if len(predictions) > 0 else 0.0
        )

        return AnomalyDetectionResult(
            pole_id=pole_id,
            time_series_name=time_series_name,
            method=self.method,
            predictions=predictions,
            scores=scores,
            n_anomalies=n_anomalies,
            anomaly_rate=anomaly_rate,
            mean_score=float(scores.mean()),
            max_score=float(scores.max()),
            min_score=float(scores.min()),
            timestamps=timestamps,
            parameters=self.method_params,
        )

    def detect_time_series_anomalies(
        self,
        time_series: np.ndarray,
        timestamps: Optional[pd.DatetimeIndex] = None,
        pole_id: str = "",
        time_series_name: str = "time_series",
    ) -> AnomalyDetectionResult:
        """
        Detect anomalies in time series data with specialized time series methods.

        Args:
            time_series: 1D array of time series values
            timestamps: Optional timestamps for the data points
            pole_id: Identifier for the pole
            time_series_name: Name of the time series

        Returns:
            AnomalyDetectionResult with detection results
        """
        # For time series, use appropriate method
        if self.method in [
            AnomalyDetectionMethod.WAVELET,
            AnomalyDetectionMethod.SEASONAL_BASELINE,
            AnomalyDetectionMethod.LSTM,
            AnomalyDetectionMethod.PYTORCH_AUTOENCODER,
        ]:
            # These methods work directly with 1D time series
            if isinstance(time_series, pd.Series):
                data = time_series.values
            else:
                data = np.asarray(time_series)

            # Special handling for seasonal baseline (requires DataFrame)
            if self.method == AnomalyDetectionMethod.SEASONAL_BASELINE:
                if timestamps is None:
                    timestamps = pd.date_range(
                        start="2020-01-01", periods=len(data), freq="D"
                    )
                df = pd.DataFrame({"date": timestamps, "value": data})
                self.detector.fit(df, date_col="date", value_col="value")
                predictions = self.detector.predict(
                    df, date_col="date", value_col="value"
                )
                # Convert predictions to scores (simple binary scoring)
                scores = np.where(predictions == -1, 1.0, 0.0)
            else:
                self.detector.fit(data)
                predictions, scores = self.detector.fit_predict(data)

        else:
            # Use standard detection
            return self.detect_anomalies(
                time_series, pole_id, time_series_name, timestamps
            )

        # Calculate statistics
        n_anomalies = int((predictions == -1).sum())
        anomaly_rate = (
            (n_anomalies / len(predictions) * 100) if len(predictions) > 0 else 0.0
        )

        return AnomalyDetectionResult(
            pole_id=pole_id,
            time_series_name=time_series_name,
            method=self.method,
            predictions=predictions,
            scores=scores,
            n_anomalies=n_anomalies,
            anomaly_rate=anomaly_rate,
            mean_score=float(scores.mean()),
            max_score=float(scores.max()),
            min_score=float(scores.min()),
            timestamps=timestamps,
            parameters=self.method_params,
        )


class EnsembleAnomalyDetector:
    """Ensemble anomaly detector combining multiple methods."""

    def __init__(
        self,
        methods: Optional[List[AnomalyDetectionMethod]] = None,
        method_params: Optional[Dict] = None,
    ):
        """
        Initialize ensemble detector.

        Args:
            methods: List of detection methods to use (default: statistical + ML methods)
            method_params: Parameters for each method
        """
        if methods is None:
            methods = [
                AnomalyDetectionMethod.ISOLATION_FOREST,
                AnomalyDetectionMethod.LOF,
                AnomalyDetectionMethod.WAVELET,
            ]

        self.methods = methods
        self.method_params = method_params or {}
        self.detectors = []

        for method in methods:
            try:
                detector = AnomalyDetector(
                    method=method,
                    method_params=self.method_params.get(method.value, {}),
                )
                self.detectors.append((method, detector))
            except Exception as e:
                logger.warning(f"Could not initialize {method.value} detector: {e}")

        if not self.detectors:
            raise ValueError("No detectors could be initialized")

    def detect_ensemble(
        self,
        time_series: np.ndarray,
        timestamps: Optional[pd.DatetimeIndex] = None,
        pole_id: str = "",
        time_series_name: str = "time_series",
        consensus_threshold: float = 0.5,  # Fraction of detectors that must agree
    ) -> EnsembleAnomalyResult:
        """
        Detect anomalies using ensemble of methods.

        Args:
            time_series: 1D array of time series values
            timestamps: Optional timestamps
            pole_id: Identifier for the pole
            time_series_name: Name of the time series
            consensus_threshold: Fraction of detectors that must flag anomaly

        Returns:
            EnsembleAnomalyResult with individual and consensus results
        """
        individual_results = {}
        all_predictions = []
        all_scores = []

        for method, detector in self.detectors:
            try:
                result = detector.detect_time_series_anomalies(
                    time_series, timestamps, pole_id, time_series_name
                )
                individual_results[method.value] = result
                all_predictions.append(result.predictions)
                all_scores.append(result.scores)
            except Exception as e:
                logger.warning(f"Error in {method.value} detection: {e}")
                continue

        if not individual_results:
            raise ValueError("No detectors produced results")

        # Calculate consensus
        predictions_array = np.array(all_predictions)
        scores_array = np.array(all_scores)

        # Consensus: anomaly if fraction of detectors agree
        n_detectors = len(predictions_array)
        anomaly_votes = (predictions_array == -1).sum(axis=0)
        consensus_predictions = np.where(
            anomaly_votes >= (n_detectors * consensus_threshold), -1, 1
        )

        # Consensus scores: average of normalized scores
        # Normalize scores to [0, 1] range
        normalized_scores = []
        for scores in scores_array:
            if scores.max() > scores.min():
                normalized = (scores - scores.min()) / (scores.max() - scores.min())
            else:
                normalized = scores
            normalized_scores.append(normalized)

        consensus_scores = np.mean(normalized_scores, axis=0)

        n_anomalies = int((consensus_predictions == -1).sum())
        anomaly_rate = (
            (n_anomalies / len(consensus_predictions) * 100)
            if len(consensus_predictions) > 0
            else 0.0
        )

        # Calculate agreement between detector pairs
        detector_agreement = {}
        method_names = list(individual_results.keys())
        for i, method1 in enumerate(method_names):
            for method2 in method_names[i + 1 :]:
                pred1 = individual_results[method1].predictions
                pred2 = individual_results[method2].predictions
                agreement = (pred1 == pred2).mean() * 100
                key = f"{method1}_{method2}"
                detector_agreement[key] = float(agreement)

        return EnsembleAnomalyResult(
            pole_id=pole_id,
            time_series_name=time_series_name,
            individual_results=individual_results,
            consensus_predictions=consensus_predictions,
            consensus_scores=consensus_scores,
            n_anomalies=n_anomalies,
            anomaly_rate=anomaly_rate,
            detector_agreement=detector_agreement,
        )
