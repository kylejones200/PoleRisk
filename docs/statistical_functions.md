# Statistical Functions Reference

This document provides detailed documentation for the statistical functions available in the polerisk package. These functions are optimized with Rust for better performance and include fallback Python implementations.

## Table of Contents
- [Overview](#overview)
- [Function Reference](#function-reference)
  - [Root Mean Square Error (RMSE)](#root-mean-square-error-rmse)
  - [Mean Absolute Error (MAE)](#mean-absolute-error-mae)
  - [Correlation Coefficient](#correlation-coefficient)
  - [Bias](#bias)
  - [Unbiased Root Mean Square Error (ubRMSE)](#unbiased-root-mean-square-error-ubrmse)
- [Usage Examples](#usage-examples)
- [Performance Considerations](#performance-considerations)
- [Mathematical Formulations](#mathematical-formulations)

## Overview

The polerisk package provides several statistical functions for analyzing and comparing soil moisture data. These functions are:

1. **RMSE (Root Mean Square Error)**: Measures the average magnitude of the errors between predicted and observed values.
2. **MAE (Mean Absolute Error)**: Measures the average absolute difference between predicted and observed values.
3. **Correlation Coefficient**: Measures the linear relationship between two datasets.
4. **Bias**: Measures the average difference between predicted and observed values.
5. **ubRMSE (Unbiased RMSE)**: Measures the precision of predictions after removing systematic bias.

## Function Reference

### Root Mean Square Error (RMSE)

```python
def calculate_rmse(x: Union[np.ndarray, list], y: Union[np.ndarray, list]) -> float
```

**Description**:
RMSE is a measure of the differences between values predicted by a model and the values observed. It represents the square root of the second sample moment of the differences between predicted values and observed values.

**Parameters**:
- `x`: Reference/observed values (ground truth)
- `y`: Predicted/estimated values

**Returns**:
- RMSE value (non-negative float, where 0.0 indicates perfect prediction)

**Example**:
```python
x = [1.0, 2.0, 3.0, 4.0, 5.0]
y = [1.1, 2.1, 2.9, 4.1, 5.0]
rmse = calculate_rmse(x, y)  # ~0.1414
```

### Mean Absolute Error (MAE)

```python
def calculate_mae(x: Union[np.ndarray, list], y: Union[np.ndarray, list]) -> float
```

**Description**:
MAE measures the average magnitude of the errors in a set of predictions, without considering their direction. It's the average over the test sample of the absolute differences between prediction and actual observation.

**Parameters**:
- `x`: Reference/observed values (ground truth)
- `y`: Predicted/estimated values

**Returns**:
- MAE value (non-negative float, where 0.0 indicates perfect prediction)

**Example**:
```python
x = [3, -0.5, 2, 7]
y = [2.5, 0.0, 2, 8]
mae = calculate_mae(x, y)  # 0.5
```

### Correlation Coefficient

```python
def calculate_correlation(x: Union[np.ndarray, list], y: Union[np.ndarray, list]) -> float
```

**Description**:
Calculates the Pearson correlation coefficient between two arrays, which measures the linear relationship between them. The correlation coefficient ranges from -1 to 1, where 1 means perfect positive correlation, -1 means perfect negative correlation, and 0 means no linear correlation.

**Parameters**:
- `x`: First array of values
- `y`: Second array of values (must be same length as x)

**Returns**:
- Correlation coefficient (float between -1.0 and 1.0)

**Example**:
```python
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]  # Perfect positive correlation
corr = calculate_correlation(x, y)  # 1.0
```

### Bias

```python
def calculate_bias(x: Union[np.ndarray, list], y: Union[np.ndarray, list]) -> float
```

**Description**:
Measures the average difference between the predicted values and the reference values. Positive bias indicates systematic over-prediction, while negative bias indicates under-prediction.

**Parameters**:
- `x`: Reference/observed values (ground truth)
- `y`: Predicted/estimated values

**Returns**:
- Bias value (float, can be positive, negative, or zero)

**Example**:
```python
x = [1, 2, 3, 4, 5]  # Observed
y = [1.1, 2.1, 3.1, 4.1, 5.1]  # Predicted (systematically higher by 0.1)
bias = calculate_bias(x, y)  # ~0.1
```

### Unbiased Root Mean Square Error (ubRMSE)

```python
def calculate_ubrmse(x: Union[np.ndarray, list], y: Union[np.ndarray, list]) -> float
```

**Description**:
Measures the precision of predictions after removing the impact of systematic bias. This is particularly useful when you want to assess the random component of the error separately from the systematic component.

**Parameters**:
- `x`: Reference/observed values (ground truth)
- `y`: Predicted/estimated values

**Returns**:
- ubRMSE value (non-negative float)

**Example**:
```python
x = [1, 2, 3, 4, 5]  # Observed
y = [1.1, 2.1, 3.1, 4.1, 5.1]  # Predicted (systematic offset)
ubrmse = calculate_ubrmse(x, y)  # Very small (just random error)
```

## Usage Examples

### Basic Usage

```python
import numpy as np
from polerisk.analysis import (
    calculate_rmse,
    calculate_mae,
    calculate_correlation,
    calculate_bias,
    calculate_ubrmse
)

# Generate sample data
np.random.seed(42)
x = np.random.rand(100) * 10  # Observed values
y = x + np.random.normal(0, 0.5, 100)  # Predicted values with some noise

# Calculate metrics
rmse = calculate_rmse(x, y)
mae = calculate_mae(x, y)
corr = calculate_correlation(x, y)
bias = calculate_bias(x, y)
ubrmse = calculate_ubrmse(x, y)

print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"Correlation: {corr:.4f}")
print(f"Bias: {bias:.4f}")
print(f"ubRMSE: {ubrmse:.4f}")
```

### Working with Time Series Data

```python
import pandas as pd
import numpy as np
from soilmoisture.analysis import calculate_rmse

# Load data (example with pandas)
df = pd.read_csv('soil_moisture_data.csv')

# Calculate RMSE for each location
results = []
for location in df['location'].unique():
    mask = df['location'] == location
    rmse = calculate_rmse(
        df.loc[mask, 'observed_sm'],
        df.loc[mask, 'predicted_sm']
    )
    results.append({'location': location, 'rmse': rmse})

results_df = pd.DataFrame(results)
print(results_df)
```

## Performance Considerations

1. **Rust vs Python**:
   - The Rust implementation is significantly faster than pure Python (5-15x speedup)
   - The package automatically uses Rust if available, with fallback to Python

2. **Array Types**:
   - For best performance, use NumPy arrays instead of Python lists
   - Ensure data is in the correct type (float64) before passing to functions

3. **Batch Processing**:
   - For very large datasets, process data in chunks to reduce memory usage
   - Consider using Dask or similar for out-of-core computations

## Mathematical Formulations

### RMSE
\[\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}\]

### MAE
\[\text{MAE} = \frac{1}{n}\sum_{i=1}^{n} |y_i - \hat{y}_i|\]

### Correlation Coefficient
\[\rho_{X,Y} = \frac{\text{cov}(X,Y)}{\sigma_X \sigma_Y}\]

### Bias
\[\text{Bias} = \frac{1}{n}\sum_{i=1}^{n} (\hat{y}_i - y_i)\]

### ubRMSE
\[\text{ubRMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n} \left[(y_i - \bar{y}) - (x_i - \bar{x})\right]^2}\]
where \(\bar{x}\) and \(\bar{y}\) are the means of x and y respectively.
