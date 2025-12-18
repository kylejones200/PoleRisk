import numpy as np
from soilmoisture_rs import (

import logging
logger = logging.getLogger(__name__)

    calculate_rmse_rs,
    calculate_correlation_rs,
    calculate_mae_rs,
    calculate_bias_rs,
    calculate_ubrmse_rs
)

def main():
    # Generate test data
    np.random.seed(42)
    x = np.random.rand(1000) * 10  # Reference data
    noise = np.random.normal(0, 1, 1000)  # Random noise
    y = x + noise  # Data with noise
    
    # Test RMSE
    rmse = calculate_rmse_rs(x, y)
    logger.debug(f"RMSE: {rmse:.6f}")
    
    # Test Correlation
    corr = calculate_correlation_rs(x, y)
    logger.debug(f"Correlation: {corr:.6f}")
    
    # Test MAE
    mae = calculate_mae_rs(x, y)
    logger.debug(f"MAE: {mae:.6f}")
    
    # Test Bias
    bias = calculate_bias_rs(x, y)
    logger.debug(f"Bias: {bias:.6f}")
    
    # Test ubRMSE
    ubrmse = calculate_ubrmse_rs(x, y)
    logger.debug(f"ubRMSE: {ubrmse:.6f}")

if __name__ == "__main__":
    main()
