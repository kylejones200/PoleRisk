"""
Example: Using ts2net and signalplot for network-based time series analysis
in pole health assessment.

This example demonstrates how to:
1. Convert soil moisture time series to network representations
2. Extract network features for pattern detection
3. Compare poles using multivariate network analysis
4. Visualize results using signalplot
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from polerisk.pole_health.ts2net_integration import (
    TS2NetAnalyzer,
    NetworkVisualizer,
    NetworkMethod
)


def generate_sample_soil_moisture_data(n_points: int = 500, pole_id: str = "POLE_001", 
                                       trend: float = 0.0, noise: float = 0.1) -> pd.DataFrame:
    """Generate sample soil moisture time series data."""
    dates = pd.date_range(start='2023-01-01', periods=n_points, freq='D')
    
    # Generate realistic soil moisture pattern with seasonality
    t = np.arange(n_points)
    seasonal = 0.3 * np.sin(2 * np.pi * t / 365) + 0.2 * np.cos(4 * np.pi * t / 365)
    trend_component = trend * t / n_points
    noise_component = np.random.normal(0, noise, n_points)
    
    # Combine components (soil moisture typically 0.1-0.5)
    moisture = 0.3 + seasonal + trend_component + noise_component
    moisture = np.clip(moisture, 0.1, 0.6)  # Clip to realistic range
    
    return pd.DataFrame({
        'date': dates,
        'pole_id': pole_id,
        'moisture_content': moisture
    })


def example_1_single_pole_analysis():
    """Example 1: Analyze a single pole's soil moisture time series."""
    print("=" * 80)
    print("Example 1: Single Pole Network Analysis")
    print("=" * 80)
    
    # Generate sample data
    df = generate_sample_soil_moisture_data(n_points=365, pole_id="POLE_001")
    
    # Initialize analyzer
    analyzer = TS2NetAnalyzer()
    
    # Analyze time series
    analysis = analyzer.analyze_time_series(
        pole_id="POLE_001",
        time_series=df['moisture_content'].values,
        time_series_name="soil_moisture",
        timestamps=df['date'],
        methods=[NetworkMethod.HVG, NetworkMethod.NVG, NetworkMethod.RECURRENCE]
    )
    
    # Print results
    print(f"\nPole ID: {analysis.pole_id}")
    print(f"Time Series: {analysis.time_series_name}")
    print(f"Data Points: {len(analysis.time_series_data)}")
    
    if analysis.hvg_features:
        print(f"\nHVG Features:")
        print(f"  Nodes: {analysis.hvg_features.n_nodes}")
        print(f"  Edges: {analysis.hvg_features.n_edges}")
        print(f"  Average Degree: {analysis.hvg_features.avg_degree:.2f}")
        print(f"  Density: {analysis.hvg_features.density:.4f}")
        if analysis.hvg_features.clustering_coefficient:
            print(f"  Clustering Coefficient: {analysis.hvg_features.clustering_coefficient:.4f}")
    
    if analysis.nvg_features:
        print(f"\nNVG Features:")
        print(f"  Nodes: {analysis.nvg_features.n_nodes}")
        print(f"  Edges: {analysis.nvg_features.n_edges}")
        print(f"  Average Degree: {analysis.nvg_features.avg_degree:.2f}")
        print(f"  Density: {analysis.nvg_features.density:.4f}")
    
    print(f"\nAnomaly Score: {analysis.anomaly_score:.4f}")
    print(f"Is Anomalous: {analysis.is_anomalous}")
    
    # Visualize
    visualizer = NetworkVisualizer()
    fig = visualizer.plot_time_series_with_network_features(
        analysis,
        save_path="Output/ts2net_single_pole_analysis.png"
    )
    print(f"\nVisualization saved to: Output/ts2net_single_pole_analysis.png")
    
    return analysis


def example_2_pole_comparison():
    """Example 2: Compare multiple poles using network features."""
    print("\n" + "=" * 80)
    print("Example 2: Multi-Pole Comparison")
    print("=" * 80)
    
    analyzer = TS2NetAnalyzer()
    
    # Generate data for multiple poles with different characteristics
    poles_data = {
        "POLE_001": generate_sample_soil_moisture_data(n_points=365, pole_id="POLE_001", trend=0.0),
        "POLE_002": generate_sample_soil_moisture_data(n_points=365, pole_id="POLE_002", trend=0.1),  # Increasing trend
        "POLE_003": generate_sample_soil_moisture_data(n_points=365, pole_id="POLE_003", trend=-0.05),  # Decreasing trend
        "POLE_004": generate_sample_soil_moisture_data(n_points=365, pole_id="POLE_004", noise=0.3),  # High noise (problematic)
    }
    
    analyses = []
    for pole_id, df in poles_data.items():
        analysis = analyzer.analyze_time_series(
            pole_id=pole_id,
            time_series=df['moisture_content'].values,
            time_series_name="soil_moisture",
            timestamps=df['date'],
            methods=[NetworkMethod.HVG]
        )
        analyses.append(analysis)
    
    # Compare to a reference (healthy pole)
    reference = analyses[0]  # Use first pole as reference
    
    print("\nComparison to Reference (POLE_001):")
    print("-" * 80)
    for analysis in analyses[1:]:
        similarity = analyzer.compare_to_reference(analysis, reference, method=NetworkMethod.HVG)
        print(f"{analysis.pole_id}: Similarity = {similarity:.4f}, Anomaly = {analysis.is_anomalous}")
    
    # Detect anomalies
    analyses = analyzer.detect_anomalous_patterns(analyses)
    
    print("\nAnomaly Detection Results:")
    print("-" * 80)
    for analysis in analyses:
        print(f"{analysis.pole_id}: Score = {analysis.anomaly_score:.4f}, Anomalous = {analysis.is_anomalous}")
    
    return analyses


def example_3_multivariate_network_analysis():
    """Example 3: Multivariate network analysis of multiple poles."""
    print("\n" + "=" * 80)
    print("Example 3: Multivariate Network Analysis")
    print("=" * 80)
    
    analyzer = TS2NetAnalyzer()
    
    # Generate time series for multiple poles
    pole_time_series = {}
    for i in range(10):
        pole_id = f"POLE_{i+1:03d}"
        df = generate_sample_soil_moisture_data(
            n_points=365,
            pole_id=pole_id,
            trend=np.random.uniform(-0.1, 0.1),
            noise=np.random.uniform(0.05, 0.2)
        )
        pole_time_series[pole_id] = df['moisture_content'].values
    
    # Analyze as multivariate network
    result = analyzer.analyze_multivariate_poles(
        pole_time_series,
        distance_method='dtw',  # Dynamic Time Warping
        k=3  # 3 nearest neighbors
    )
    
    if 'error' in result:
        print(f"Error: {result['error']}")
        return None
    
    print(f"\nMultivariate Network Results:")
    print(f"  Number of Poles: {result['n_nodes']}")
    print(f"  Number of Edges: {result['n_edges']}")
    print(f"  Average Degree: {result['avg_degree']:.2f}")
    
    # Visualize distance matrix
    visualizer = NetworkVisualizer()
    fig = visualizer.plot_multivariate_network(
        result['distance_matrix'],
        result['pole_ids'],
        save_path="Output/ts2net_multivariate_network.png"
    )
    print(f"\nMultivariate network visualization saved to: Output/ts2net_multivariate_network.png")
    
    return result


def example_4_integration_with_pole_health():
    """Example 4: Integration with existing pole health assessment."""
    print("\n" + "=" * 80)
    print("Example 4: Integration with Pole Health Assessment")
    print("=" * 80)
    
    from polerisk.pole_health.assessment import PoleHealthAssessment
    from polerisk.pole_health.pole_data import PoleInfo, SoilSample
    from datetime import datetime
    
    # Create pole info
    pole = PoleInfo(
        pole_id="POLE_ANALYSIS_001",
        latitude=40.7128,
        longitude=-74.0060,
        install_date=datetime(2010, 1, 1),
        material="southern_pine",
        height_ft=45.0
    )
    
    # Generate historical soil samples
    samples = []
    base_date = datetime(2022, 1, 1)
    for i in range(12):  # Monthly samples for 1 year
        sample_date = base_date + timedelta(days=30*i)
        moisture = 0.3 + 0.1 * np.sin(2 * np.pi * i / 12) + np.random.normal(0, 0.05)
        
        sample = SoilSample(
            pole_id=pole.pole_id,
            sample_date=sample_date,
            depth_inches=12.0,
            moisture_content=moisture,
            ph=6.5,
            bulk_density=1.4
        )
        samples.append(sample)
    
    # Perform standard health assessment
    assessor = PoleHealthAssessment()
    health_metrics = assessor.assess_pole_health(pole, samples)
    
    print(f"\nStandard Assessment:")
    print(f"  Health Score: {health_metrics.overall_health_score:.2f}")
    print(f"  Risk Score: {health_metrics.moisture_risk:.4f}")
    
    # Now add network-based analysis
    analyzer = TS2NetAnalyzer()
    
    # Extract moisture time series
    moisture_values = [s.moisture_content for s in sorted(samples, key=lambda x: x.sample_date)]
    timestamps = pd.DatetimeIndex([s.sample_date for s in sorted(samples, key=lambda x: x.sample_date)])
    
    # Analyze with network methods
    network_analysis = analyzer.analyze_time_series(
        pole_id=pole.pole_id,
        time_series=np.array(moisture_values),
        time_series_name="soil_moisture",
        timestamps=timestamps,
        methods=[NetworkMethod.HVG]
    )
    
    print(f"\nNetwork-Based Analysis:")
    if network_analysis.hvg_features:
        print(f"  Network Density: {network_analysis.hvg_features.density:.4f}")
        print(f"  Average Degree: {network_analysis.hvg_features.avg_degree:.2f}")
        print(f"  Anomaly Score: {network_analysis.anomaly_score:.4f}")
        print(f"  Is Anomalous: {network_analysis.is_anomalous}")
    
    # Combine insights
    if network_analysis.is_anomalous:
        print(f"\n⚠️  Warning: Network analysis indicates anomalous pattern in soil moisture time series.")
        print(f"   This may indicate unusual degradation patterns or environmental stress.")
    
    return health_metrics, network_analysis


if __name__ == "__main__":
    # Ensure output directory exists
    from pathlib import Path
    Path("Output").mkdir(exist_ok=True)
    
    print("\n" + "=" * 80)
    print("ts2net and signalplot Integration Examples")
    print("=" * 80)
    print("\nThese examples demonstrate network-based time series analysis")
    print("for utility pole health assessment.\n")
    
    # Run examples
    try:
        example_1_single_pole_analysis()
        example_2_pole_comparison()
        example_3_multivariate_network_analysis()
        example_4_integration_with_pole_health()
        
        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)
        
    except ImportError as e:
        print(f"\n❌ Error: Required library not installed: {e}")
        print("\nPlease install required dependencies:")
        print("  pip install ts2net signalplot")
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()

