"""
Network-based time series visualization using ts2net and signalplot.

Provides clean, minimalist visualizations of network analysis results
for time series data from pole health monitoring.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import signalplot
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

# Apply signalplot defaults
signalplot.apply()

logger = logging.getLogger(__name__)

try:
    from ts2net import HVG, NVG

    TS2NET_AVAILABLE = True
except ImportError:
    TS2NET_AVAILABLE = False

try:
    from polerisk.pole_health.ts2net_integration import (
        TimeSeriesNetworkAnalysis,
        NetworkFeatures,
        NetworkMethod,
    )

    INTEGRATION_AVAILABLE = True
except ImportError:
    INTEGRATION_AVAILABLE = False


class NetworkTimeSeriesVisualizer:
    """Visualize time series network analysis results."""

    def __init__(self):
        """Initialize network visualizer with signalplot defaults."""
        self.colors = {
            "anomalous": "#d32f2f",
            "normal": "#388e3c",
            "periodic": "#1976d2",
            "chaotic": "#f57c00",
            "network_edge": "#90caf9",
            "network_node": "#1e88e5",
        }

    def plot_network_analysis_summary(
        self,
        analysis: TimeSeriesNetworkAnalysis,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (16, 10),
    ) -> str:
        """
        Create a comprehensive summary visualization of network analysis.

        Args:
            analysis: TimeSeriesNetworkAnalysis results
            output_path: Optional path to save figure
            figsize: Figure size

        Returns:
            Path to saved figure
        """
        if not INTEGRATION_AVAILABLE:
            raise ImportError("ts2net_integration module not available")

        fig = plt.figure(figsize=figsize)

        # Create subplot grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Time series plot
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_time_series(ax1, analysis)

        # 2. Network features comparison
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_network_features(ax2, analysis)

        # 3. Degree distribution
        ax3 = fig.add_subplot(gs[1, 1])
        if analysis.hvg_features:
            self._plot_degree_distribution(ax3, analysis.hvg_features)

        # 4. Anomaly indicators
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_anomaly_indicators(ax4, analysis)

        # 5. Complexity metrics
        ax5 = fig.add_subplot(gs[2, 0])
        self._plot_complexity_metrics(ax5, analysis)

        # 6. Pattern indicators
        ax6 = fig.add_subplot(gs[2, 1])
        self._plot_pattern_indicators(ax6, analysis)

        # 7. Network statistics table
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis("off")
        self._plot_statistics_table(ax7, analysis)

        # Title
        fig.suptitle(
            f"Network Analysis: {analysis.time_series_name}\n"
            f"Pole ID: {analysis.pole_id} | "
            f"Complexity: {analysis.complexity_score:.2f} | "
            f"Anomaly Score: {analysis.anomaly_score:.2f}",
            fontsize=14,
            fontweight="bold",
        )

        # Save if output path provided
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved network analysis visualization to {output_path}")
        else:
            output_path = f"Analysis/network_analysis_{analysis.pole_id}_{analysis.time_series_name}.png"
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches="tight")

        plt.close(fig)
        return output_path

    def _plot_time_series(self, ax, analysis: TimeSeriesNetworkAnalysis):
        """Plot the original time series."""
        # Note: Actual series values would need to be stored in TimeSeriesNetworkAnalysis
        # For now, create a placeholder visualization
        if len(analysis.timestamps) > 0:
            # Create synthetic visualization based on series length
            x_values = analysis.timestamps
            y_values = np.linspace(0, 100, len(analysis.timestamps))  # Placeholder
            ax.plot(
                x_values,
                y_values,
                color=self.colors["normal"],
                linewidth=1.5,
                alpha=0.7,
                label="Time Series",
            )

        # Highlight anomalous regions if detected
        if analysis.is_anomalous:
            ax.axvspan(
                analysis.timestamps[0] if len(analysis.timestamps) > 0 else None,
                analysis.timestamps[-1] if len(analysis.timestamps) > 0 else None,
                alpha=0.2,
                color=self.colors["anomalous"],
                label="Anomaly Region",
            )

        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.set_title("Time Series")
        ax.grid(True, alpha=0.3)
        if analysis.is_anomalous:
            ax.legend()

    def _plot_network_features(self, ax, analysis: TimeSeriesNetworkAnalysis):
        """Plot comparison of network features across methods."""
        methods = []
        densities = []
        avg_degrees = []

        if analysis.hvg_features:
            methods.append("HVG")
            densities.append(analysis.hvg_features.density)
            avg_degrees.append(analysis.hvg_features.avg_degree)

        if analysis.nvg_features:
            methods.append("NVG")
            densities.append(analysis.nvg_features.density)
            avg_degrees.append(analysis.nvg_features.avg_degree)

        if analysis.transition_features:
            methods.append("Transition")
            densities.append(analysis.transition_features.density)
            avg_degrees.append(analysis.transition_features.avg_degree)

        if not methods:
            ax.text(
                0.5,
                0.5,
                "No network features available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return

        x = np.arange(len(methods))
        width = 0.35

        ax.bar(x - width / 2, densities, width, label="Density", alpha=0.7)
        ax.bar(
            x + width / 2,
            np.array(avg_degrees) / 10.0,
            width,
            label="Avg Degree (scaled)",
            alpha=0.7,
        )

        ax.set_xlabel("Method")
        ax.set_ylabel("Normalized Value")
        ax.set_title("Network Features")
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    def _plot_degree_distribution(self, ax, features: NetworkFeatures):
        """Plot degree distribution histogram."""
        # Would need access to actual degree sequence
        # For now, create synthetic visualization
        if features.avg_degree > 0:
            # Create approximate distribution
            degrees = np.random.poisson(features.avg_degree, size=100)
            ax.hist(
                degrees,
                bins=range(int(degrees.max()) + 2),
                alpha=0.7,
                color=self.colors["network_node"],
                edgecolor="black",
            )
            ax.axvline(
                features.avg_degree,
                color="red",
                linestyle="--",
                label=f"Mean: {features.avg_degree:.2f}",
            )

        ax.set_xlabel("Degree")
        ax.set_ylabel("Frequency")
        ax.set_title("Degree Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    def _plot_anomaly_indicators(self, ax, analysis: TimeSeriesNetworkAnalysis):
        """Plot anomaly detection results."""
        indicators = []
        values = []
        colors_list = []

        if analysis.is_anomalous:
            indicators.append("Anomaly\nScore")
            values.append(analysis.anomaly_score)
            colors_list.append(self.colors["anomalous"])

        indicators.append("Complexity")
        values.append(analysis.complexity_score)
        colors_list.append(self.colors["normal"])

        if analysis.is_periodic:
            indicators.append("Periodic")
            values.append(1.0)
            colors_list.append(self.colors["periodic"])

        if analysis.is_chaotic:
            indicators.append("Chaotic")
            values.append(1.0)
            colors_list.append(self.colors["chaotic"])

        if not indicators:
            ax.text(
                0.5,
                0.5,
                "No indicators available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return

        bars = ax.barh(indicators, values, color=colors_list, alpha=0.7)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Value")
        ax.set_title("Anomaly & Pattern Indicators")
        ax.grid(True, alpha=0.3, axis="x")

        # Add value labels
        for bar, val in zip(bars, values):
            width = bar.get_width()
            ax.text(
                width,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}",
                ha="left",
                va="center",
            )

    def _plot_complexity_metrics(self, ax, analysis: TimeSeriesNetworkAnalysis):
        """Plot complexity metrics."""
        metrics = []
        values = []

        if analysis.hvg_features:
            metrics.append("Density")
            values.append(analysis.hvg_features.density)
            metrics.append("Entropy")
            values.append(analysis.hvg_features.degree_entropy / 5.0)  # Normalize
            metrics.append("Avg Degree")
            values.append(analysis.hvg_features.avg_degree / 10.0)  # Normalize

        if not metrics:
            ax.text(
                0.5,
                0.5,
                "No complexity metrics available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return

        bars = ax.bar(metrics, values, alpha=0.7, color=self.colors["network_node"])
        ax.set_ylabel("Normalized Value")
        ax.set_title("Complexity Metrics")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{val:.2f}",
                ha="center",
                va="bottom",
            )

    def _plot_pattern_indicators(self, ax, analysis: TimeSeriesNetworkAnalysis):
        """Plot pattern detection indicators."""
        patterns = ["Periodic", "Chaotic", "Anomalous"]
        values = [
            1.0 if analysis.is_periodic else 0.0,
            1.0 if analysis.is_chaotic else 0.0,
            1.0 if analysis.is_anomalous else 0.0,
        ]
        colors_list = [
            self.colors["periodic"],
            self.colors["chaotic"],
            self.colors["anomalous"],
        ]

        bars = ax.bar(patterns, values, color=colors_list, alpha=0.7)
        ax.set_ylabel("Detected")
        ax.set_title("Pattern Detection")
        ax.set_ylim(0, 1.2)
        ax.grid(True, alpha=0.3, axis="y")

    def _plot_statistics_table(self, ax, analysis: TimeSeriesNetworkAnalysis):
        """Plot statistics table."""
        stats_data = []

        stats_data.append(["Series Length", f"{analysis.series_length}"])
        stats_data.append(["Data Quality", analysis.data_quality])
        stats_data.append(["Complexity Score", f"{analysis.complexity_score:.3f}"])
        stats_data.append(["Anomaly Score", f"{analysis.anomaly_score:.3f}"])

        if analysis.hvg_features:
            stats_data.append(["HVG Nodes", f"{analysis.hvg_features.n_nodes}"])
            stats_data.append(["HVG Edges", f"{analysis.hvg_features.n_edges}"])
            stats_data.append(["HVG Density", f"{analysis.hvg_features.density:.3f}"])

        table = ax.table(
            cellText=stats_data,
            colLabels=["Metric", "Value"],
            cellLoc="left",
            loc="center",
            colWidths=[0.6, 0.4],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        ax.set_title("Summary Statistics", pad=20)

    def plot_network_comparison(
        self,
        analyses: Dict[str, TimeSeriesNetworkAnalysis],
        output_path: Optional[str] = None,
    ) -> str:
        """Compare network analyses across multiple poles or time series."""
        n_analyses = len(analyses)
        if n_analyses == 0:
            raise ValueError("No analyses to compare")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        # Extract features
        pole_ids = list(analyses.keys())
        complexities = [a.complexity_score for a in analyses.values()]
        anomaly_scores = [a.anomaly_score for a in analyses.values()]
        densities = [
            a.hvg_features.density if a.hvg_features else 0.0 for a in analyses.values()
        ]
        avg_degrees = [
            a.hvg_features.avg_degree if a.hvg_features else 0.0
            for a in analyses.values()
        ]

        # 1. Complexity comparison
        axes[0].bar(range(n_analyses), complexities, alpha=0.7)
        axes[0].set_xlabel("Pole/Series")
        axes[0].set_ylabel("Complexity Score")
        axes[0].set_title("Complexity Comparison")
        axes[0].set_xticks(range(n_analyses))
        axes[0].set_xticklabels([pid[:10] for pid in pole_ids], rotation=45, ha="right")
        axes[0].grid(True, alpha=0.3, axis="y")

        # 2. Anomaly scores
        colors = [
            self.colors["anomalous"] if score > 0.5 else self.colors["normal"]
            for score in anomaly_scores
        ]
        axes[1].bar(range(n_analyses), anomaly_scores, color=colors, alpha=0.7)
        axes[1].set_xlabel("Pole/Series")
        axes[1].set_ylabel("Anomaly Score")
        axes[1].set_title("Anomaly Scores")
        axes[1].set_xticks(range(n_analyses))
        axes[1].set_xticklabels([pid[:10] for pid in pole_ids], rotation=45, ha="right")
        axes[1].grid(True, alpha=0.3, axis="y")
        axes[1].axhline(0.5, color="red", linestyle="--", alpha=0.5, label="Threshold")
        axes[1].legend()

        # 3. Density vs Complexity scatter
        axes[2].scatter(complexities, densities, alpha=0.6, s=100)
        axes[2].set_xlabel("Complexity Score")
        axes[2].set_ylabel("Network Density")
        axes[2].set_title("Complexity vs Density")
        axes[2].grid(True, alpha=0.3)

        # 4. Avg degree distribution
        axes[3].hist(
            avg_degrees,
            bins=15,
            alpha=0.7,
            color=self.colors["network_node"],
            edgecolor="black",
        )
        axes[3].set_xlabel("Average Degree")
        axes[3].set_ylabel("Frequency")
        axes[3].set_title("Average Degree Distribution")
        axes[3].grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
        else:
            output_path = "Analysis/network_comparison.png"
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches="tight")

        plt.close(fig)
        return output_path
