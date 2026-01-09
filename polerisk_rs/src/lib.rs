use numpy::PyArray1;
use pyo3::prelude::*;
use ndarray::ArrayView1;

/// Calculate Root Mean Square Error (RMSE) between two arrays
#[pyfunction]
fn calculate_rmse_rs(_py: Python, x: &PyArray1<f64>, y: &PyArray1<f64>) -> PyResult<f64> {
    let x_arr: ArrayView1<f64>;
    let y_arr: ArrayView1<f64>;
    
    // SAFETY: We know these arrays are valid for the duration of this function
    unsafe {
        x_arr = x.as_array();
        y_arr = y.as_array();
    }
    
    if x_arr.len() != y_arr.len() {
        return Err(pyo3::exceptions::PyValueError::new_err("Input arrays must have the same length"));
    }
    
    let mse = x_arr.iter()
        .zip(y_arr.iter())
        .map(|(xi, yi)| (xi - yi).powi(2))
        .sum::<f64>() / x_arr.len() as f64;
        
    Ok(mse.sqrt())
}

/// Calculate correlation coefficient between two arrays
#[pyfunction]
fn calculate_correlation_rs(_py: Python, x: &PyArray1<f64>, y: &PyArray1<f64>) -> PyResult<f64> {
    let x_arr: ArrayView1<f64>;
    let y_arr: ArrayView1<f64>;
    
    // SAFETY: We know these arrays are valid for the duration of this function
    unsafe {
        x_arr = x.as_array();
        y_arr = y.as_array();
    }
    
    if x_arr.len() != y_arr.len() {
        return Err(pyo3::exceptions::PyValueError::new_err("Input arrays must have the same length"));
    }
    
    let n = x_arr.len() as f64;
    let sum_x: f64 = x_arr.sum();
    let sum_y: f64 = y_arr.sum();
    
    let sum_xy: f64 = x_arr.iter().zip(y_arr.iter()).map(|(xi, yi)| xi * yi).sum();
    let sum_x2: f64 = x_arr.iter().map(|xi| xi * xi).sum();
    let sum_y2: f64 = y_arr.iter().map(|yi| yi * yi).sum();
    
    let numerator = n * sum_xy - sum_x * sum_y;
    let denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();
    
    if denominator == 0.0 {
        return Ok(0.0);
    }
    
    Ok(numerator / denominator)
}

/// Calculate Mean Absolute Error (MAE) between two arrays
#[pyfunction]
fn calculate_mae_rs(_py: Python, x: &PyArray1<f64>, y: &PyArray1<f64>) -> PyResult<f64> {
    let x_arr: ArrayView1<f64>;
    let y_arr: ArrayView1<f64>;
    
    // SAFETY: We know these arrays are valid for the duration of this function
    unsafe {
        x_arr = x.as_array();
        y_arr = y.as_array();
    }
    
    if x_arr.len() != y_arr.len() {
        return Err(pyo3::exceptions::PyValueError::new_err("Input arrays must have the same length"));
    }
    
    let mae = x_arr.iter()
        .zip(y_arr.iter())
        .map(|(xi, yi)| (xi - yi).abs())
        .sum::<f64>() / x_arr.len() as f64;
        
    Ok(mae)
}

/// Calculate bias between two arrays
#[pyfunction]
fn calculate_bias_rs(_py: Python, x: &PyArray1<f64>, y: &PyArray1<f64>) -> PyResult<f64> {
    let x_arr: ArrayView1<f64>;
    let y_arr: ArrayView1<f64>;
    
    // SAFETY: We know these arrays are valid for the duration of this function
    unsafe {
        x_arr = x.as_array();
        y_arr = y.as_array();
    }
    
    if x_arr.len() != y_arr.len() {
        return Err(pyo3::exceptions::PyValueError::new_err("Input arrays must have the same length"));
    }
    
    let bias = y_arr.iter()
        .zip(x_arr.iter())
        .map(|(yi, xi)| yi - xi)
        .sum::<f64>() / x_arr.len() as f64;
        
    Ok(bias)
}

/// Calculate Unbiased Root Mean Square Error (ubRMSE)
#[pyfunction]
fn calculate_ubrmse_rs(py: Python, x: &PyArray1<f64>, y: &PyArray1<f64>) -> PyResult<f64> {
    let x_arr: ArrayView1<f64>;
    let y_arr: ArrayView1<f64>;
    
    // SAFETY: We know these arrays are valid for the duration of this function
    unsafe {
        x_arr = x.as_array();
        y_arr = y.as_array();
    }
    
    if x_arr.len() != y_arr.len() {
        return Err(pyo3::exceptions::PyValueError::new_err("Input arrays must have the same length"));
    }
    
    let bias = calculate_bias_rs(py, x, y)?;
    let mse = x_arr.iter()
        .zip(y_arr.iter())
        .map(|(xi, yi)| {
            let diff = yi - xi - bias;
            diff * diff
        })
        .sum::<f64>() / x_arr.len() as f64;
        
    Ok(mse.sqrt())
}

/// A Python module implemented in Rust
#[pymodule]
fn polerisk_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(calculate_rmse_rs, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_correlation_rs, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_mae_rs, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_bias_rs, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_ubrmse_rs, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use pyo3::Python;
    use numpy::PyArray1;

    fn create_test_arrays() -> (Vec<f64>, Vec<f64>) {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.1, 2.1, 2.9, 4.1, 4.9];
        (x, y)
    }

    #[test]
    fn test_calculate_rmse() {
        let (x, y) = create_test_arrays();
        Python::with_gil(|py| {
            let x_arr = PyArray1::from_vec(py, x.clone());
            let y_arr = PyArray1::from_vec(py, y);
            let rmse = calculate_rmse_rs(py, x_arr, y_arr).unwrap();
            assert_relative_eq!(rmse, 0.1414, epsilon = 1e-4);
        });
    }

    #[test]
    fn test_calculate_correlation() {
        let (x, y) = create_test_arrays();
        Python::with_gil(|py| {
            let x_arr = PyArray1::from_vec(py, x.clone());
            let y_arr = PyArray1::from_vec(py, y);
            let corr = calculate_correlation_rs(py, x_arr, y_arr).unwrap();
            assert_relative_eq!(corr, 0.99939, epsilon = 1e-4);
        });
    }

    #[test]
    fn test_calculate_mae() {
        let (x, y) = create_test_arrays();
        Python::with_gil(|py| {
            let x_arr = PyArray1::from_vec(py, x.clone());
            let y_arr = PyArray1::from_vec(py, y);
            let mae = calculate_mae_rs(py, x_arr, y_arr).unwrap();
            assert_relative_eq!(mae, 0.12, epsilon = 1e-10);
        });
    }

    #[test]
    fn test_calculate_bias() {
        let (x, y) = create_test_arrays();
        Python::with_gil(|py| {
            let x_arr = PyArray1::from_vec(py, x.clone());
            let y_arr = PyArray1::from_vec(py, y);
            let bias = calculate_bias_rs(py, x_arr, y_arr).unwrap();
            assert_relative_eq!(bias, 0.0, epsilon = 1e-10);
        });
    }

    #[test]
    fn test_calculate_ubrmse() {
        let (x, y) = create_test_arrays();
        Python::with_gil(|py| {
            let x_arr = PyArray1::from_vec(py, x.clone());
            let y_arr = PyArray1::from_vec(py, y);
            let ubrmse = calculate_ubrmse_rs(py, x_arr, y_arr).unwrap();
            assert_relative_eq!(ubrmse, 0.1414, epsilon = 1e-4);
        });
    }

    #[test]
    fn test_error_handling() {
        Python::with_gil(|py| {
            let x = PyArray1::from_vec(py, vec![1.0, 2.0]);
            let y = PyArray1::from_vec(py, vec![1.0]);
            
            assert!(calculate_rmse_rs(py, x, y).is_err());
            assert!(calculate_correlation_rs(py, x, y).is_err());
            assert!(calculate_mae_rs(py, x, y).is_err());
            assert!(calculate_bias_rs(py, x, y).is_err());
            assert!(calculate_ubrmse_rs(py, x, y).is_err());
        });
    }
}
