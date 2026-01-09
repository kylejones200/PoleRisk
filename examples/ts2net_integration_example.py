#!/usr/bin/env python3
"""
Example demonstrating ts2net and signalplot integration for pole health analysis.

This example shows how to use ts2net for network-based time series analysis
of pole health metrics and soil moisture data, with signalplot for clean
visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import signalplot
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from polerisk.pole_health.ts2net_integration import (
    TS2NetAnalyzer,
    NetworkMethod,
    TimeSeriesNetworkAnalysis
)
from polerisk.visualization.network_visualization import NetworkTimeSeriesVisualizer

# Apply signalplot defaults
signalplot.apply()

def generate_sample_soil_moisture_data(days: int = 365) -> pd.Series:
    """Generate sample soil moisture time series with patterns."""
    dates = pd.date_range(start=datetime.now() - timedelta(days=days), periods=days, freq='D')
    
    # Create realistic soil moisture patterns
    # Base trend (increasing moisture over time - potential issue)
    trend = np.linspace(0.3, 0.7, days)
    
    # Seasonal pattern (higher in winter/spring)
    seasonal = 0.2 * np.sin(2 * np.pi * np.arange(days) / 365.25)
    
    # Random noise
    noise = np.random.normal(0, 0.05, days)
    
    # Add some anomalies
    anomaly_indices = np.random.choice(days, size=5, replace=False)
    anomalies = np.zeros(days)
    anomalies[anomaly_indices] = np.random.uniform(0.3, 0.5, 5)
    
    moisture = trend + seasonal + noise + anomalies
    moisture = np.clip(moisture, 0.0, 1.0)
    
    return pd.Series(moisture, index=dates, name='soil_moisture')


def generate_sample_health_score_data(days: int = 365) -> pd.Series:
    """Generate sample pole health score time series."""
    dates = pd.date_range(start=datetime.now() - timedelta(days=days), periods=days, freq='D')
    
    # Health score decreasing over time (degradation)
    initial_score = 85.0
    final_score = 45.0
    trend = np.linspace(initial_score, final_score, days)
    
    # Random fluctuations
    noise = np.random.normal(0, 3.0, days)
    
    # Step decreases (maintenance events)
    step_decreases = np.zeros(days)
    step_indices = np.random.choice(days, size=3, replace=False)
    step_decreases[step_indices] = -10
    
    score = trend + noise + step_decreases
    score = np.clip(score, 0.0, 100.0)
    
    return pd.Series(score, index=dates, name='health_score')


def analyze_pole_time_series():
    """Demonstrate ts2net analysis on pole health data."""
    print("=" * 60)
    print("TS2NET INTEGRATION EXAMPLE")
    print("=" * 60)
    
    # Generate sample data
    print("\n1. Generating sample time series data...")
    soil_moisture = generate_sample_soil_moisture_data(days=180)
    health_score = generate_sample_health_score_data(days=180)
    
    print(f"   Soil moisture series: {len(soil_moisture)} points")
    print(f"   Health score series: {len(health_score)} points")
    
    # Initialize analyzer
    print("\n2. Initializing TS2Net analyzer...")
    analyzer = TS2NetAnalyzer()
    
    # Analyze soil moisture
    print("\n3. Analyzing soil moisture time series...")
    moisture_analysis = analyzer.analyze_time_series(
        pole_id="POLE_001",
        time_series=soil_moisture.values,
        time_series_name="soil_moisture",
        timestamps=soil_moisture.index,
        methods=[NetworkMethod.HVG, NetworkMethod.TRANSITION]
    )
    
    print(f"   Network Analysis Results:")
    print(f"   - Data quality: {moisture_analysis.data_quality}")
    print(f"   - Complexity score: {moisture_analysis.complexity_score:.3f}")
    print(f"   - Anomaly detected: {moisture_analysis.is_anomalous}")
    print(f"   - Anomaly score: {moisture_analysis.anomaly_score:.3f}")
    if moisture_analysis.anomaly_reasons:
        print(f"   - Anomaly reasons: {', '.join(moisture_analysis.anomaly_reasons)}")
    print(f"   - Periodic pattern: {moisture_analysis.is_periodic}")
    print(f"   - Chaotic pattern: {moisture_analysis.is_chaotic}")
    
    if moisture_analysis.hvg_features:
        print(f"\n   HVG Features:")
        print(f"   - Nodes: {moisture_analysis.hvg_features.n_nodes}")
        print(f"   - Edges: {moisture_analysis.hvg_features.n_edges}")
        print(f"   - Density: {moisture_analysis.hvg_features.density:.3f}")
        print(f"   - Avg degree: {moisture_analysis.hvg_features.avg_degree:.2f}")
        print(f"   - Degree entropy: {moisture_analysis.hvg_features.degree_entropy:.2f}")
    
    # Analyze health score
    print("\n4. Analyzing health score time series...")
    health_analysis = analyzer.analyze_time_series(
        pole_id="POLE_001",
        time_series=health_score.values,
        time_series_name="health_score",
        timestamps=health_score.index,
        methods=[NetworkMethod.HVG, NetworkMethod.TRANSITION]
    )
    
    print(f"   Network Analysis Results:")
    print(f"   - Data quality: {health_analysis.data_quality}")
    print(f"   - Complexity score: {health_analysis.complexity_score:.3f}")
    print(f"   - Anomaly detected: {health_analysis.is_anomalous}")
    print(f"   - Anomaly score: {health_analysis.anomaly_score:.3f}")
    
    if health_analysis.hvg_features:
        print(f"\n   HVG Features:")
        print(f"   - Nodes: {health_analysis.hvg_features.n_nodes}")
        print(f"   - Edges: {health_analysis.hvg_features.n_edges}")
        print(f"   - Density: {health_analysis.hvg_features.density:.3f}")
        print(f"   - Avg degree: {health_analysis.hvg_features.avg_degree:.2f}")
    
    # Create visualizations
    print("\n5. Creating visualizations with signalplot...")
    visualizer = NetworkTimeSeriesVisualizer()
    
    # Save network analysis visualizations
    output_dir = Path("Analysis/ts2net_examples")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    moisture_viz_path = visualizer.plot_network_analysis_summary(
        moisture_analysis,
        output_path=str(output_dir / "soil_moisture_network_analysis.png")
    )
    print(f"   Saved: {moisture_viz_path}")
    
    health_viz_path = visualizer.plot_network_analysis_summary(
        health_analysis,
        output_path=str(output_dir / "health_score_network_analysis.png")
    )
    print(f"   Saved: {health_viz_path}")
    
    # Comparison visualization
    comparisons = {
        "POLE_001_moisture": moisture_analysis,
        "POLE_001_health": health_analysis
    }
    comparison_path = visualizer.plot_network_comparison(
        comparisons,
        output_path=str(output_dir / "network_comparison.png")
    )
    print(f"   Saved: {comparison_path}")
    
    # Create simple time series plot with signalplot
    print("\n6. Creating time series plots with signalplot defaults...")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Soil moisture plot
    axes[0].plot(soil_moisture.index, soil_moisture.values, linewidth=1.5, alpha=0.8)
    axes[0].set_title('Soil Moisture Time Series', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Moisture Content')
    axes[0].grid(True, alpha=0.3)
    
    # Health score plot
    axes[1].plot(health_score.index, health_score.values, linewidth=1.5, 
                alpha=0.8, color='orange')
    axes[1].set_title('Pole Health Score Time Series', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Health Score')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(50, color='red', linestyle='--', alpha=0.5, label='Risk Threshold')
    axes[1].legend()
    
    plt.tight_layout()
    ts_plot_path = output_dir / "time_series_plots.png"
    plt.savefig(ts_plot_path, dpi=300, bbox_inches='tight')
    print(f"   Saved: {ts_plot_path}")
    plt.close()
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nAll outputs saved to: {output_dir}")
    print("\nKey Insights:")
    print(f"- Soil moisture complexity: {moisture_analysis.complexity_score:.3f}")
    print(f"- Health score complexity: {health_analysis.complexity_score:.3f}")
    if moisture_analysis.is_anomalous:
        print(f"- Soil moisture anomalies detected - investigate pole conditions")
    if health_analysis.complexity_score > 0.7:
        print(f"- High health score complexity - degradation pattern detected")


if __name__ == "__main__":
    try:
        analyze_pole_time_series()
    except ImportError as e:
        print(f"\nError: {e}")
        print("\nPlease install required packages:")
        print("  pip install ts2net signalplot")
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()

