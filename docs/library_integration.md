# Library Integration: ts2net and signalplot

This document describes the integration of two key libraries into the polerisk project:

- **ts2net**: Time series to network conversion for pattern analysis
- **signalplot**: Minimalist plotting defaults for Matplotlib

## Overview

Both libraries have been integrated to enhance time series analysis and visualization capabilities for utility pole health assessment.

## ts2net Integration

### Purpose

The ts2net library enables network-based analysis of time series data, allowing us to:
- Detect complex patterns in soil moisture time series
- Identify anomalies in pole health metrics
- Analyze degradation trends using network topology
- Compare patterns across multiple poles

### Integration Points

1. **Time Series Analysis Module** (`polerisk/pole_health/ts2net_integration.py`)
   - `TS2NetAnalyzer`: Main analyzer class
   - `TimeSeriesNetworkAnalysis`: Complete analysis results
   - Support for multiple network methods:
     - HVG (Horizontal Visibility Graph)
     - NVG (Natural Visibility Graph)
     - Recurrence Networks
     - Transition Networks

2. **Soil Condition Analyzer** (`polerisk/pole_health/assessment.py`)
   - Enhanced `analyze_temporal_trends()` method with network analysis option
   - Pattern detection for moisture trends
   - Anomaly detection based on network features

3. **Visualization** (`polerisk/visualization/network_visualization.py`)
   - Network analysis visualization
   - Comparison plots across poles
   - Anomaly indicators

### Usage Example

```python
from polerisk.pole_health.ts2net_integration import (
    TS2NetAnalyzer, NetworkMethod
)
import numpy as np
import pandas as pd

# Initialize analyzer
analyzer = TS2NetAnalyzer()

# Analyze soil moisture time series
moisture_data = np.array([...])  # Your time series data
timestamps = pd.date_range(...)

analysis = analyzer.analyze_time_series(
    pole_id="POLE_001",
    time_series=moisture_data,
    time_series_name="soil_moisture",
    timestamps=timestamps,
    methods=[NetworkMethod.HVG, NetworkMethod.TRANSITION]
)

# Check results
print(f"Complexity: {analysis.complexity_score}")
print(f"Anomaly detected: {analysis.is_anomalous}")
print(f"Periodic pattern: {analysis.is_periodic}")
```

### Features

- **Anomaly Detection**: Identifies unusual patterns in time series
- **Complexity Scoring**: Measures series complexity using network metrics
- **Pattern Detection**: Identifies periodic and chaotic patterns
- **BSTS Integration**: Optional decomposition for structural analysis
- **Memory Efficient**: Handles large time series with optimized modes

### References

- GitHub: https://github.com/kylejones200/ts2net
- Documentation: See ts2net README for full API documentation

## signalplot Integration

### Purpose

signalplot provides minimalist, clean plotting defaults for Matplotlib that prioritize data clarity over decoration.

### Integration Points

1. **All Visualization Modules**
   - Applied globally via `signalplot.apply()` in visualization modules
   - Used in:
     - `polerisk/visualization/pole_health_viz.py`
     - `polerisk/visualization/network_visualization.py`
     - `examples/ts2net_integration_example.py`

2. **Default Styling**
   - Clean, minimal plots
   - Focus on data presentation
   - Professional appearance for reports

### Usage

signalplot is applied automatically in visualization modules:

```python
import signalplot

# Apply defaults (done automatically in visualization modules)
signalplot.apply()

# Your plotting code - will use signalplot defaults
import matplotlib.pyplot as plt
plt.plot(x, y)
plt.show()
```

### Benefits

- **Consistent Styling**: All plots use the same clean style
- **Report Ready**: Professional appearance suitable for executive reports
- **Data Focus**: Minimal decoration, maximum clarity
- **Less Code**: No need to manually style every plot

### References

- GitHub: https://github.com/kylejones200/signalplot

## Combined Usage Example

See `examples/ts2net_integration_example.py` for a complete example demonstrating:

1. Generating sample pole health time series
2. Analyzing with ts2net
3. Visualizing with signalplot-enhanced matplotlib
4. Creating network analysis visualizations

Run the example:

```bash
python examples/ts2net_integration_example.py
```

## Installation

Both libraries are included in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install ts2net signalplot
```

From GitHub (development versions):

```bash
pip install git+https://github.com/kylejones200/ts2net.git
pip install git+https://github.com/kylejones200/signalplot.git
```

## Dependencies

### ts2net Dependencies
- numpy
- scipy (optional, for advanced features)
- statsmodels (optional, for BSTS features)
- numba (optional, for performance acceleration)
- networkx (optional, for network analysis)

### signalplot Dependencies
- matplotlib

All dependencies are handled automatically by pip.

## Performance Considerations

### ts2net

- **Small series** (< 10k points): Use all methods with default settings
- **Medium series** (10k - 100k points): 
  - Use `output="degrees"` for memory efficiency
  - Set `limit` parameter for NVG
- **Large series** (> 100k points):
  - Use `output="degrees"` or `output="stats"`
  - Always set `limit` for NVG
  - Use k-NN for recurrence networks

### signalplot

- Minimal performance impact
- Applied once at module import
- No ongoing performance overhead

## Troubleshooting

### ts2net Import Errors

If you see `ts2net not available`:

1. Check installation: `pip list | grep ts2net`
2. Reinstall: `pip install --upgrade ts2net`
3. Check Python version (requires 3.12+)

### Network Analysis Fails

- Ensure time series has at least 10 points
- Check for NaN values in time series
- Verify timestamps are properly formatted
- Use smaller `limit` parameter for large series

### signalplot Not Applied

- Ensure `signalplot.apply()` is called before plotting
- Check import: `import signalplot`
- Verify signalplot is installed: `pip list | grep signalplot`

## Future Enhancements

Potential future improvements:

1. **Batch Processing**: Parallel analysis of multiple poles
2. **Real-time Analysis**: Streaming time series analysis
3. **Custom Network Methods**: Domain-specific network construction
4. **Interactive Visualizations**: Plotly integration with signalplot styling
5. **Model Integration**: Use network features in ML models

## References

- ts2net: https://github.com/kylejones200/ts2net
- signalplot: https://github.com/kylejones200/signalplot
- Example script: `examples/ts2net_integration_example.py`

