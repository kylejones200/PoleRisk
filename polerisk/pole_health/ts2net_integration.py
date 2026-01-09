"""
Integration of ts2net library for time series network analysis.

Leverages ts2net for network-based pattern detection in pole health time series,
soil moisture data, and environmental sensor readings. Provides anomaly detection
and pattern analysis capabilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)

try:
    from ts2net import HVG, NVG, RecurrenceNetwork, TransitionNetwork, build_network
    from ts2net.bsts import features as bsts_features, BSTSSpec

    TS2NET_AVAILABLE = True
except ImportError:
    TS2NET_AVAILABLE = False
    logger.warning("ts2net not available. Install with: pip install ts2net")


class NetworkMethod(Enum):
    """Available network construction methods."""

    HVG = "hvg"  # Horizontal Visibility Graph
    NVG = "nvg"  # Natural Visibility Graph
    RECURRENCE = "recurrence"  # Recurrence network
    TRANSITION = "transition"  # Transition network


@dataclass
class NetworkFeatures:
    """Features extracted from network analysis."""

    method: NetworkMethod
    n_nodes: int
    n_edges: int
    density: float
    avg_degree: float
    max_degree: int
    min_degree: int
    degree_entropy: float  # Entropy of degree distribution
    clustering_coefficient: Optional[float] = None
    path_length: Optional[float] = None
    small_world_propensity: Optional[float] = None


@dataclass
class TimeSeriesNetworkAnalysis:
    """Complete network analysis results for a time series."""

    pole_id: str
    time_series_name: str
    timestamps: pd.DatetimeIndex

    # Network features by method
    hvg_features: Optional[NetworkFeatures] = None
    nvg_features: Optional[NetworkFeatures] = None
    recurrence_features: Optional[NetworkFeatures] = None
    transition_features: Optional[NetworkFeatures] = None

    # Anomaly detection
    is_anomalous: bool = False
    anomaly_score: float = 0.0
    anomaly_reasons: List[str] = field(default_factory=list)

    # Pattern indicators
    is_periodic: bool = False
    is_chaotic: bool = False
    complexity_score: float = 0.0

    # BSTS decomposition results (if used)
    structural_stats: Optional[Dict] = None
    residual_network_stats: Optional[Dict] = None

    # Analysis metadata
    analysis_date: datetime = field(default_factory=datetime.now)
    series_length: int = 0
    data_quality: str = "good"  # good, fair, poor


class TS2NetAnalyzer:
    """Analyzer for time series using ts2net network methods."""

    def __init__(self):
        """Initialize ts2net analyzer."""
        if not TS2NET_AVAILABLE:
            logger.error("ts2net not available. Network analysis will not work.")
            raise ImportError("ts2net is required. Install with: pip install ts2net")

        # Default parameters for different methods
        self.default_params = {
            NetworkMethod.HVG: {"weighted": False, "limit": None, "output": "edges"},
            NetworkMethod.NVG: {
                "weighted": False,
                "limit": 5000,  # Limit for large series
                "max_edges": 1_000_000,
                "output": "edges",
            },
            NetworkMethod.RECURRENCE: {
                "m": 3,  # Embedding dimension
                "tau": 1,  # Time delay
                "rule": "knn",  # k-nearest neighbors
                "k": 10,  # Number of neighbors
            },
            NetworkMethod.TRANSITION: {
                "symbolizer": "ordinal",
                "order": 3,  # Ordinal pattern order
            },
        }

        # Baseline statistics for anomaly detection (can be updated from historical data)
        self.baseline_stats: Dict[str, Dict] = {}

    def analyze_time_series(
        self,
        pole_id: str,
        time_series: np.ndarray,
        time_series_name: str,
        timestamps: Optional[pd.DatetimeIndex] = None,
        methods: List[NetworkMethod] = None,
        use_bsts: bool = False,
        bsts_params: Optional[Dict] = None,
    ) -> TimeSeriesNetworkAnalysis:
        """
        Analyze a time series using network methods.

        Args:
            pole_id: Pole identifier
            time_series: Time series data as numpy array
            time_series_name: Name/type of time series (e.g., "soil_moisture", "health_score")
            timestamps: Optional timestamps for the series
            methods: List of network methods to use (default: [HVG, TRANSITION])
            use_bsts: If True, use BSTS decomposition first
            bsts_params: Parameters for BSTS decomposition

        Returns:
            TimeSeriesNetworkAnalysis with network features
        """
        if methods is None:
            methods = [NetworkMethod.HVG, NetworkMethod.TRANSITION]

        # Prepare timestamps
        if timestamps is None:
            timestamps = pd.date_range(
                start=datetime.now() - pd.Timedelta(days=len(time_series)),
                periods=len(time_series),
                freq="D",
            )

        # Check data quality
        data_quality = self._assess_data_quality(time_series)

        # Apply BSTS decomposition if requested
        residual_series = time_series
        structural_stats = None
        residual_network_stats = None

        if use_bsts:
            try:
                bsts_result = self._apply_bsts_decomposition(
                    time_series, timestamps, bsts_params
                )
                residual_series = bsts_result["residual"]
                structural_stats = bsts_result["structural_stats"]
                residual_network_stats = bsts_result["residual_network_stats"]
            except Exception as e:
                logger.warning(
                    f"BSTS decomposition failed: {e}. Using original series."
                )
                residual_series = time_series

        # Analyze with each method
        hvg_features = None
        nvg_features = None
        recurrence_features = None
        transition_features = None

        for method in methods:
            try:
                if method == NetworkMethod.HVG:
                    hvg_features = self._analyze_with_hvg(residual_series)
                elif method == NetworkMethod.NVG:
                    nvg_features = self._analyze_with_nvg(residual_series)
                elif method == NetworkMethod.RECURRENCE:
                    recurrence_features = self._analyze_with_recurrence(residual_series)
                elif method == NetworkMethod.TRANSITION:
                    transition_features = self._analyze_with_transition(residual_series)
            except Exception as e:
                logger.warning(f"Network analysis with {method.value} failed: {e}")

        # Detect anomalies
        is_anomalous, anomaly_score, anomaly_reasons = self._detect_anomalies(
            pole_id,
            time_series_name,
            hvg_features,
            nvg_features,
            transition_features,
            recurrence_features,
        )

        # Detect patterns
        is_periodic, is_chaotic, complexity_score = self._detect_patterns(
            hvg_features, nvg_features, transition_features, recurrence_features
        )

        return TimeSeriesNetworkAnalysis(
            pole_id=pole_id,
            time_series_name=time_series_name,
            timestamps=timestamps,
            hvg_features=hvg_features,
            nvg_features=nvg_features,
            recurrence_features=recurrence_features,
            transition_features=transition_features,
            is_anomalous=is_anomalous,
            anomaly_score=anomaly_score,
            anomaly_reasons=anomaly_reasons,
            is_periodic=is_periodic,
            is_chaotic=is_chaotic,
            complexity_score=complexity_score,
            structural_stats=structural_stats,
            residual_network_stats=residual_network_stats,
            series_length=len(time_series),
            data_quality=data_quality,
        )

    def _analyze_with_hvg(self, series: np.ndarray) -> NetworkFeatures:
        """Analyze time series with Horizontal Visibility Graph."""
        params = self.default_params[NetworkMethod.HVG]

        # Adjust output mode based on series length
        if len(series) > 100000:
            params = params.copy()
            params["output"] = "degrees"

        hvg = HVG(**params)
        hvg.build(series)

        return self._extract_network_features(hvg, NetworkMethod.HVG)

    def _analyze_with_nvg(self, series: np.ndarray) -> NetworkFeatures:
        """Analyze time series with Natural Visibility Graph."""
        params = self.default_params[NetworkMethod.NVG].copy()

        # Adjust limit based on series length
        if len(series) > 10000:
            params["limit"] = min(5000, len(series) // 2)
            params["output"] = "degrees"

        nvg = NVG(**params)
        nvg.build(series)

        return self._extract_network_features(nvg, NetworkMethod.NVG)

    def _analyze_with_recurrence(self, series: np.ndarray) -> NetworkFeatures:
        """Analyze time series with Recurrence Network."""
        params = self.default_params[NetworkMethod.RECURRENCE]

        # Use k-NN for large series
        if len(series) > 10000:
            params = params.copy()
            params["rule"] = "knn"
            params["k"] = min(30, len(series) // 100)

        rn = RecurrenceNetwork(**params)
        rn.build(series)

        return self._extract_network_features(rn, NetworkMethod.RECURRENCE)

    def _analyze_with_transition(self, series: np.ndarray) -> NetworkFeatures:
        """Analyze time series with Transition Network."""
        params = self.default_params[NetworkMethod.TRANSITION]

        tn = TransitionNetwork(**params)
        tn.build(series)

        return self._extract_network_features(tn, NetworkMethod.TRANSITION)

    def _extract_network_features(
        self, network, method: NetworkMethod
    ) -> NetworkFeatures:
        """Extract features from a network object."""
        n_nodes = network.n_nodes
        n_edges = network.n_edges

        # Calculate density
        max_edges = n_nodes * (n_nodes - 1) / 2
        density = n_edges / max_edges if max_edges > 0 else 0.0

        # Get degree sequence
        degrees = network.degree_sequence()
        avg_degree = float(np.mean(degrees)) if len(degrees) > 0 else 0.0
        max_degree = int(np.max(degrees)) if len(degrees) > 0 else 0
        min_degree = int(np.min(degrees)) if len(degrees) > 0 else 0

        # Calculate degree entropy
        if len(degrees) > 0:
            degree_counts = np.bincount(degrees)
            degree_probs = degree_counts / len(degrees)
            degree_probs = degree_probs[degree_probs > 0]  # Remove zeros
            degree_entropy = float(-np.sum(degree_probs * np.log2(degree_probs)))
        else:
            degree_entropy = 0.0

        # Optional: calculate clustering coefficient and path length (requires NetworkX)
        clustering = None
        path_length = None
        small_world = None

        try:
            if hasattr(network, "as_networkx"):
                G = network.as_networkx()
                if G.number_of_nodes() > 0 and G.number_of_edges() > 0:
                    import networkx as nx

                    clustering = float(nx.average_clustering(G))

                    if nx.is_connected(G):
                        path_length = float(nx.average_shortest_path_length(G))

                        # Small-world propensity (simplified)
                        # Compare to random graph with same number of nodes/edges
                        n = G.number_of_nodes()
                        m = G.number_of_edges()
                        random_clustering = (2 * m) / (n * (n - 1)) if n > 1 else 0
                        random_path = (
                            np.log(n) / np.log(avg_degree) if avg_degree > 1 else 0
                        )

                        if random_clustering > 0 and random_path > 0:
                            small_world = (
                                clustering / random_clustering
                                if random_clustering > 0
                                else None
                            )
        except Exception as e:
            logger.debug(f"Could not calculate advanced network metrics: {e}")

        return NetworkFeatures(
            method=method,
            n_nodes=n_nodes,
            n_edges=n_edges,
            density=density,
            avg_degree=avg_degree,
            max_degree=max_degree,
            min_degree=min_degree,
            degree_entropy=degree_entropy,
            clustering_coefficient=clustering,
            path_length=path_length,
            small_world_propensity=small_world,
        )

    def _apply_bsts_decomposition(
        self,
        series: np.ndarray,
        timestamps: pd.DatetimeIndex,
        params: Optional[Dict] = None,
    ) -> Dict:
        """Apply BSTS decomposition using ts2net's BSTS features."""
        if params is None:
            params = {
                "level": True,
                "trend": True,
                "seasonal_periods": [24, 168],  # Daily and weekly for hourly data
            }

        spec = BSTSSpec(**params)

        # Use ts2net's BSTS features function
        result = bsts_features(series, methods=["hvg", "transition"], bsts=spec)

        return {
            "residual": result.residual_network_stats.get("residual", series),
            "structural_stats": result.structural_stats,
            "residual_network_stats": result.residual_network_stats,
        }

    def _detect_anomalies(
        self,
        pole_id: str,
        series_name: str,
        hvg: Optional[NetworkFeatures],
        nvg: Optional[NetworkFeatures],
        transition: Optional[NetworkFeatures],
        recurrence: Optional[NetworkFeatures],
    ) -> Tuple[bool, float, List[str]]:
        """Detect anomalies based on network features."""
        is_anomalous = False
        anomaly_score = 0.0
        reasons = []

        # Check against baseline if available
        baseline_key = f"{series_name}_baseline"
        baseline = self.baseline_stats.get(baseline_key)

        if baseline and hvg:
            # Check density anomaly
            if hvg.density > baseline.get("hvg_density_mean", 0.5) + 2 * baseline.get(
                "hvg_density_std", 0.1
            ):
                is_anomalous = True
                anomaly_score += 0.3
                reasons.append("Unusually high network density")

            # Check degree distribution anomaly
            if hvg.avg_degree > baseline.get(
                "hvg_avg_degree_mean", 4.0
            ) + 2 * baseline.get("hvg_avg_degree_std", 1.0):
                is_anomalous = True
                anomaly_score += 0.2
                reasons.append("Unusual degree distribution")

        # Check for extreme values
        if hvg:
            if hvg.density > 0.95:
                is_anomalous = True
                anomaly_score += 0.2
                reasons.append("Extremely dense network (potential data quality issue)")

            if hvg.avg_degree < 1.0:
                is_anomalous = True
                anomaly_score += 0.2
                reasons.append("Very sparse network (potential data gaps)")

        if transition and transition.n_edges < 5:
            is_anomalous = True
            anomaly_score += 0.1
            reasons.append("Very few transitions detected")

        return is_anomalous, min(1.0, anomaly_score), reasons

    def _detect_patterns(
        self,
        hvg: Optional[NetworkFeatures],
        nvg: Optional[NetworkFeatures],
        transition: Optional[NetworkFeatures],
        recurrence: Optional[NetworkFeatures],
    ) -> Tuple[bool, bool, float]:
        """Detect periodic and chaotic patterns."""
        is_periodic = False
        is_chaotic = False
        complexity_score = 0.0

        if transition:
            # Periodic series tend to have specific transition patterns
            if transition.n_edges > 10 and transition.avg_degree > 2.0:
                is_periodic = True

        if hvg:
            # Chaotic series have high degree entropy
            if hvg.degree_entropy > 3.0:
                is_chaotic = True

            # Complexity score based on network properties
            complexity_score = (
                hvg.density * 0.3
                + (hvg.degree_entropy / 5.0) * 0.4  # Normalize entropy
                + (hvg.avg_degree / 10.0) * 0.3  # Normalize degree
            )
            complexity_score = min(1.0, complexity_score)

        return is_periodic, is_chaotic, complexity_score

    def _assess_data_quality(self, series: np.ndarray) -> str:
        """Assess data quality of time series."""
        # Check for missing values
        missing_ratio = np.isnan(series).sum() / len(series) if len(series) > 0 else 0.0

        # Check for constant values
        if len(series) > 0:
            unique_ratio = len(np.unique(series)) / len(series)
        else:
            unique_ratio = 0.0

        # Check for outliers (values beyond 3 standard deviations)
        if len(series) > 0 and np.nanstd(series) > 0:
            outliers = np.abs(series - np.nanmean(series)) > 3 * np.nanstd(series)
            outlier_ratio = outliers.sum() / len(series)
        else:
            outlier_ratio = 0.0

        # Assess quality
        if missing_ratio < 0.05 and unique_ratio > 0.1 and outlier_ratio < 0.05:
            return "good"
        elif missing_ratio < 0.15 and unique_ratio > 0.05 and outlier_ratio < 0.10:
            return "fair"
        else:
            return "poor"

    def update_baseline_stats(
        self, series_name: str, features_list: List[NetworkFeatures]
    ):
        """Update baseline statistics from a collection of network features."""
        if not features_list:
            return

        hvg_features = [f for f in features_list if f.method == NetworkMethod.HVG]

        if hvg_features:
            densities = [f.density for f in hvg_features]
            avg_degrees = [f.avg_degree for f in hvg_features]

            self.baseline_stats[f"{series_name}_baseline"] = {
                "hvg_density_mean": np.mean(densities),
                "hvg_density_std": np.std(densities),
                "hvg_avg_degree_mean": np.mean(avg_degrees),
                "hvg_avg_degree_std": np.std(avg_degrees),
            }


def batch_analyze_time_series(
    poles_df: pd.DataFrame,
    time_series_column: str,
    timestamps_column: Optional[str] = None,
    methods: List[NetworkMethod] = None,
) -> Dict[str, TimeSeriesNetworkAnalysis]:
    """
    Batch analyze time series for multiple poles.

    Args:
        poles_df: DataFrame with pole data and time series
        time_series_column: Column name containing time series data
        timestamps_column: Optional column name for timestamps
        methods: Network methods to use

    Returns:
        Dictionary mapping pole_id to TimeSeriesNetworkAnalysis
    """
    analyzer = TS2NetAnalyzer()
    results = {}

    for _, row in poles_df.iterrows():
        pole_id = str(row.get("pole_id", "unknown"))
        series = row.get(time_series_column)

        if series is None or len(series) < 10:
            continue

        if isinstance(series, list):
            series = np.array(series)

        timestamps = None
        if timestamps_column and timestamps_column in row:
            timestamps = pd.DatetimeIndex(row[timestamps_column])

        try:
            analysis = analyzer.analyze_time_series(
                pole_id=pole_id,
                time_series=series,
                time_series_name=time_series_column,
                timestamps=timestamps,
                methods=methods,
            )
            results[pole_id] = analysis
        except Exception as e:
            logger.warning(f"Failed to analyze time series for pole {pole_id}: {e}")

    return results
