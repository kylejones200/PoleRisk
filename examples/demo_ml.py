#!/usr/bin/env python3
"""
Demo script showcasing the new machine learning capabilities.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Import our new ML modules
from soilmoisture.ml.models import SoilMoisturePredictor, AnomalyDetector, TimeSeriesForecaster
from soilmoisture.ml.features import FeatureEngineer, create_temporal_features, select_features
from soilmoisture.analysis.statistics import calculate_rmse, calculate_correlation, calculate_bias


def create_demo_data():
    """Create realistic demo data for ML testing."""
    logger.debug("Creating synthetic soil moisture data for demo...")
    
    # Generate 2 years of daily data
    dates = pd.date_range('2021-01-01', '2022-12-31', freq='D')
    n_days = len(dates)
    
    # Create realistic seasonal pattern
    day_of_year = np.arange(1, n_days + 1) % 365
    
    # Base seasonal cycle (higher in winter, lower in summer for this example)
    seasonal_base = 0.15 + 0.1 * np.cos(2 * np.pi * day_of_year / 365 + np.pi)
    
    # Add weather variability
    np.random.seed(42)
    weather_noise = np.random.normal(0, 0.02, n_days)
    precipitation_events = np.random.exponential(0.05, n_days)
    precipitation_events[precipitation_events > 0.2] = 0  # Cap extreme values
    
    # Create in-situ data (ground truth)
    in_situ = seasonal_base + weather_noise + precipitation_events
    in_situ = np.clip(in_situ, 0.05, 0.45)  # Realistic soil moisture range
    
    # Create satellite data with some bias and noise
    satellite_bias = 0.05  # Satellite tends to overestimate
    satellite_noise = np.random.normal(0, 0.03, n_days)
    satellite = in_situ + satellite_bias + satellite_noise
    satellite = np.clip(satellite, 0.0, 0.5)
    
    # Add some missing satellite data (realistic)
    missing_indices = np.random.choice(n_days, size=int(0.1 * n_days), replace=False)
    satellite[missing_indices] = np.nan
    
    # Create temperature and precipitation for weather features
    temperature = 15 + 15 * np.cos(2 * np.pi * day_of_year / 365) + np.random.normal(0, 3, n_days)
    precipitation = precipitation_events * 100  # Convert to mm
    
    demo_data = pd.DataFrame({
        'date': dates,
        'in_situ': in_situ,
        'satellite': satellite,
        'temperature': temperature,
        'precipitation': precipitation
    })
    
    return demo_data


def demo_feature_engineering(data):
    """Demonstrate feature engineering capabilities."""
    logger.debug("\n=== Feature Engineering Demo ===")
    
    # Basic temporal features
    features_basic = create_temporal_features(data)
    logger.debug(f"Added temporal features. New columns: {len(features_basic.columns) - len(data.columns)}")
    
    # Advanced feature engineering
    fe = FeatureEngineer()
    features_advanced = fe.fit_transform(data)
    
    logger.info(f"Advanced feature engineering completed.")
    logger.debug(f"Original features: {len(data.columns)}")
    logger.debug(f"Final features: {len(features_advanced.columns)}")
    
    # Feature selection
    if 'in_situ' in features_advanced.columns:
        selected_features = select_features(
            features_advanced, 
            target_col='in_situ',
            correlation_threshold=0.05,
            max_features=20
        )
        logger.debug(f"Selected {len(selected_features)} most relevant features")
        logger.debug("Top selected features:", selected_features[:10])
    
    return features_advanced, selected_features


def demo_prediction_models(data, features, feature_cols):
    """Demonstrate different prediction models."""
    logger.debug("\n=== Soil Moisture Prediction Demo ===")
    
    # Prepare data
    X = features[feature_cols].dropna()
    y = features.loc[X.index, 'in_situ']
    
    logger.debug(f"Training on {len(X)} samples with {len(feature_cols)} features")
    
    results = {}
    
    # Test different models
    model_types = ['random_forest', 'linear', 'svm']
    
    for model_type in model_types:
        try:
            logger.debug(f"\nTraining {model_type} model...")
            predictor = SoilMoisturePredictor(model_type=model_type)
            metrics = predictor.fit(X, y, validation_split=0.2)
            
            # Make predictions on test set
            predictions = predictor.predict(X)
            
            # Calculate test metrics
            test_rmse = calculate_rmse(y, predictions)
            test_corr = calculate_correlation(y, predictions)
            test_bias = calculate_bias(y, predictions)
            
            results[model_type] = {
                'predictor': predictor,
                'val_rmse': metrics['val_rmse'],
                'val_r2': metrics['val_r2'],
                'test_rmse': test_rmse,
                'test_corr': test_corr,
                'test_bias': test_bias
            }
            
            logger.debug(f"  Validation RMSE: {metrics['val_rmse']:.4f}")
            logger.debug(f"  Test RMSE: {test_rmse:.4f}")
            logger.debug(f"  Test Correlation: {test_corr:.4f}")
            
        except Exception as e:
            logger.error(f"  Failed to train {model_type}: {e}")
    
    # Show best model
    if results:
        best_model = min(results.keys(), key=lambda x: results[x]['val_rmse'])
        logger.debug(f"\nBest model: {best_model} (RMSE: {results[best_model]['val_rmse']:.4f})")
        
        # Feature importance for tree-based models
        if best_model == 'random_forest':
            importance = results[best_model]['predictor'].get_feature_importance()
            if importance:
                logger.debug("\nTop 5 most important features:")
                sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                for feat, imp in sorted_features[:5]:
                    logger.debug(f"  {feat}: {imp:.4f}")
    
    return results


def demo_anomaly_detection(data):
    """Demonstrate anomaly detection."""
    logger.debug("\n=== Anomaly Detection Demo ===")
    
    # Use satellite data for anomaly detection
    satellite_data = data[['satellite']].dropna()
    
    if len(satellite_data) < 10:
        logger.debug("Not enough satellite data for anomaly detection")
        return None
    
    # Train detector
    detector = AnomalyDetector(method='isolation_forest')
    detector.fit(satellite_data)
    
    # Detect anomalies
    anomaly_labels, anomaly_scores = detector.detect_anomalies(satellite_data)
    
    n_anomalies = anomaly_labels.sum()
    anomaly_rate = n_anomalies / len(anomaly_labels) * 100
    
    logger.debug(f"Detected {n_anomalies} anomalies ({anomaly_rate:.1f}% of data)")
    
    # Add anomalies back to original data
    data_with_anomalies = data.copy()
    data_with_anomalies.loc[satellite_data.index, 'is_anomaly'] = anomaly_labels
    data_with_anomalies.loc[satellite_data.index, 'anomaly_score'] = anomaly_scores
    
    return data_with_anomalies


def demo_forecasting(data):
    """Demonstrate time series forecasting."""
    logger.debug("\n=== Time Series Forecasting Demo ===")
    
    # Use in-situ data for forecasting
    time_series = data['in_situ'].dropna()
    
    if len(time_series) < 50:
        logger.debug("Not enough data for reliable forecasting")
        return None
    
    # Split data: use first 80% for training, forecast the rest
    split_point = int(len(time_series) * 0.8)
    train_series = time_series.iloc[:split_point]
    test_series = time_series.iloc[split_point:]
    
    logger.debug(f"Training on {len(train_series)} points, testing on {len(test_series)} points")
    
    # Try linear model (more reliable than LSTM for demo)
    forecaster = TimeSeriesForecaster(model_type='linear', forecast_horizon=len(test_series))
    
    try:
        metrics = forecaster.fit(train_series)
        forecasts = forecaster.forecast(train_series, steps=len(test_series))
        
        # Calculate forecast accuracy
        forecast_rmse = calculate_rmse(test_series.values, forecasts)
        forecast_corr = calculate_correlation(test_series.values, forecasts)
        
        logger.debug(f"Training metrics: {metrics}")
        logger.debug(f"Forecast RMSE: {forecast_rmse:.4f}")
        logger.debug(f"Forecast correlation: {forecast_corr:.4f}")
        
        return {
            'forecaster': forecaster,
            'train_series': train_series,
            'test_series': test_series,
            'forecasts': forecasts,
            'forecast_rmse': forecast_rmse
        }
        
    except Exception as e:
        logger.error(f"Forecasting failed: {e}")
        return None


def create_demo_visualizations(data, prediction_results, anomaly_data, forecast_results):
    """Create visualizations of the ML results."""
    logger.debug("\n=== Creating Visualizations ===")
    
    os.makedirs('ML_Demo_Output', exist_ok=True)
    
    # 1. Original data visualization
    plt.figure(figsize=(15, 10))
    
    # Time series plot
    plt.subplot(2, 2, 1)
    plt.plot(data['date'], data['in_situ'], 'b-', label='In-situ', alpha=0.8)
    plt.plot(data['date'], data['satellite'], 'r--', label='Satellite', alpha=0.8)
    plt.title('Original Soil Moisture Data')
    plt.ylabel('Soil Moisture (m³/m³)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Scatter plot
    plt.subplot(2, 2, 2)
    valid_data = data.dropna(subset=['in_situ', 'satellite'])
    plt.scatter(valid_data['in_situ'], valid_data['satellite'], alpha=0.6)
    plt.plot([0, 0.5], [0, 0.5], 'k--', alpha=0.5)
    plt.xlabel('In-situ Soil Moisture')
    plt.ylabel('Satellite Soil Moisture')
    plt.title('In-situ vs Satellite')
    plt.grid(True, alpha=0.3)
    
    # Anomalies (if available)
    if anomaly_data is not None:
        plt.subplot(2, 2, 3)
        plt.plot(anomaly_data['date'], anomaly_data['satellite'], 'b-', alpha=0.7, label='Normal')
        anomalies = anomaly_data[anomaly_data.get('is_anomaly', False)]
        if len(anomalies) > 0:
            plt.scatter(anomalies['date'], anomalies['satellite'], 
                       c='red', s=30, label='Anomalies', zorder=5)
        plt.title('Anomaly Detection Results')
        plt.ylabel('Satellite Soil Moisture')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
    
    # Forecasting (if available)
    if forecast_results is not None:
        plt.subplot(2, 2, 4)
        train_dates = data.loc[forecast_results['train_series'].index, 'date']
        test_dates = data.loc[forecast_results['test_series'].index, 'date']
        
        plt.plot(train_dates, forecast_results['train_series'], 'b-', label='Training Data')
        plt.plot(test_dates, forecast_results['test_series'], 'g-', label='Actual', linewidth=2)
        plt.plot(test_dates, forecast_results['forecasts'], 'r--', label='Forecast', linewidth=2)
        plt.title('Time Series Forecasting')
        plt.ylabel('In-situ Soil Moisture')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ML_Demo_Output/ml_demo_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.debug("Visualization saved to ML_Demo_Output/ml_demo_results.png")
    
    # 2. Model comparison plot
    if prediction_results:
        plt.figure(figsize=(10, 6))
        
        models = list(prediction_results.keys())
        val_rmse = [prediction_results[m]['val_rmse'] for m in models]
        test_rmse = [prediction_results[m]['test_rmse'] for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        plt.bar(x - width/2, val_rmse, width, label='Validation RMSE', alpha=0.8)
        plt.bar(x + width/2, test_rmse, width, label='Test RMSE', alpha=0.8)
        
        plt.xlabel('Model Type')
        plt.ylabel('RMSE')
        plt.title('Model Performance Comparison')
        plt.xticks(x, models)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('ML_Demo_Output/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.debug("Model comparison saved to ML_Demo_Output/model_comparison.png")


def main():
    """Main demo function."""
    logger.debug(" Soil Moisture Machine Learning Demo ")
    logger.debug("=" * 50)
    
    # Create demo data
    data = create_demo_data()
    logger.debug(f"Generated {len(data)} days of synthetic soil moisture data")
    logger.debug(f"Date range: {data['date'].min()} to {data['date'].max()}")
    logger.debug(f"In-situ range: {data['in_situ'].min():.3f} to {data['in_situ'].max():.3f} m³/m³")
    logger.debug(f"Satellite coverage: {(~data['satellite'].isna()).mean()*100:.1f}%")
    
    # Feature engineering
    features, selected_features = demo_feature_engineering(data)
    
    # Prediction models
    prediction_results = demo_prediction_models(data, features, selected_features)
    
    # Anomaly detection
    anomaly_data = demo_anomaly_detection(data)
    
    # Forecasting
    forecast_results = demo_forecasting(data)
    
    # Create visualizations
    create_demo_visualizations(data, prediction_results, anomaly_data, forecast_results)
    
    logger.debug("\n=== Demo Summary ===")
    logger.debug(" Feature Engineering: Created temporal, statistical, and weather-proxy features")
    
    if prediction_results:
        best_model = min(prediction_results.keys(), key=lambda x: prediction_results[x]['val_rmse'])
        logger.debug(f" Prediction Models: Best model is {best_model} (RMSE: {prediction_results[best_model]['val_rmse']:.4f})")
    
    if anomaly_data is not None:
        n_anomalies = anomaly_data.get('is_anomaly', pd.Series([])).sum()
        logger.debug(f" Anomaly Detection: Identified {n_anomalies} potential anomalies")
    
    if forecast_results is not None:
        logger.debug(f" Time Series Forecasting: Achieved {forecast_results['forecast_rmse']:.4f} RMSE on test data")
    
    logger.debug(f" Visualizations: Saved to ML_Demo_Output/")
    
    # Save demo data for further experimentation
    data.to_csv('ML_Demo_Output/demo_data.csv', index=False)
    logger.debug(f" Demo data saved to ML_Demo_Output/demo_data.csv")
    
    logger.info("\n Demo completed successfully!")
    logger.debug("\nNext steps:")
    logger.debug("1. Try the CLI interface: python -m soilmoisture.ml.cli train --data-file ML_Demo_Output/demo_data.csv")
    logger.debug("2. Experiment with different model types and parameters")
    logger.debug("3. Use real soil moisture data for production models")


if __name__ == "__main__":
    main()
