"""
Ensemble forecasting combining classification and regression models.

This module implements a hybrid approach where:
- Classification model predicts direction (increase/decrease)
- Regression model predicts magnitude using classification predictions as features

This approach has been shown to outperform standard ARIMA models for time series
forecasting, especially for complex patterns and noisy data.

Based on: Refined_Ensemble_Models_For_Time_Series.md
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import logging

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller

    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ForecastResult:
    """Results from ensemble forecasting."""

    # Predictions
    predictions: np.ndarray
    actual: Optional[np.ndarray] = None

    # Metrics
    mae: Optional[float] = None  # Mean Absolute Error
    mse: Optional[float] = None  # Mean Squared Error
    rmse: Optional[float] = None  # Root Mean Squared Error
    mape: Optional[float] = None  # Mean Absolute Percentage Error
    direction_accuracy: Optional[float] = None  # Classification accuracy

    # Model details
    classification_accuracy: Optional[float] = None
    regression_mae: Optional[float] = None
    arima_mae: Optional[float] = None  # For comparison

    # Timestamps
    timestamps: Optional[pd.DatetimeIndex] = None

    # Metadata
    forecast_date: datetime = field(default_factory=datetime.now)
    n_train: int = 0
    n_test: int = 0


class EnsembleForecaster:
    """
    Ensemble forecaster combining classification and regression models.

    Uses Random Forest classifier to predict direction (up/down) and Random Forest
    regressor to predict magnitude, with classification predictions as features.
    """

    def __init__(
        self,
        n_lags: int = 2,
        test_size: float = 0.2,
        random_state: int = 42,
        n_estimators_classifier: int = 100,
        n_estimators_regressor: int = 100,
        max_depth: Optional[int] = None,
    ):
        """
        Initialize ensemble forecaster.

        Args:
            n_lags: Number of lagged features to use
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
            n_estimators_classifier: Number of trees for classifier
            n_estimators_regressor: Number of trees for regressor
            max_depth: Maximum depth of trees (None = unlimited)
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for ensemble forecasting")

        self.n_lags = n_lags
        self.test_size = test_size
        self.random_state = random_state

        self.classifier = RandomForestClassifier(
            n_estimators=n_estimators_classifier,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )

        self.regressor = RandomForestRegressor(
            n_estimators=n_estimators_regressor,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )

        self.scaler = StandardScaler()
        self.is_fitted = False
        self.arima_model = None

    def _create_features(
        self, data: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Create features for classification and regression.

        Args:
            data: Time series data

        Returns:
            Tuple of (features, direction_target, regression_target)
        """
        df = pd.DataFrame({"value": data})

        # Create lagged features
        for lag in range(1, self.n_lags + 1):
            df[f"lag_{lag}"] = df["value"].shift(lag)

        # Rate of change
        df["rate_of_change"] = df["value"].diff()

        # Moving averages
        if len(df) > 5:
            df["ma_3"] = df["value"].rolling(window=3, min_periods=1).mean()
            df["ma_5"] = df["value"].rolling(window=5, min_periods=1).mean()

        # Target for classification: 1 if next value increases, 0 if decreases
        df["direction"] = (df["value"].shift(-1) > df["value"]).astype(int)

        # Target for regression: next value
        df["next_value"] = df["value"].shift(-1)

        # Drop NaN rows
        df = df.dropna()

        if len(df) == 0:
            raise ValueError("Not enough data to create features")

        # Features (excluding targets)
        feature_cols = [
            col for col in df.columns if col not in ["direction", "next_value", "value"]
        ]
        X = df[feature_cols]

        y_class = df["direction"]
        y_reg = df["next_value"]

        return X, y_class, y_reg

    def fit(self, data: pd.Series) -> Dict:
        """
        Fit ensemble models on training data.

        Args:
            data: Time series data for training

        Returns:
            Dictionary with training metrics
        """
        if len(data) < self.n_lags + 10:
            raise ValueError(
                f"Need at least {self.n_lags + 10} data points for training"
            )

        # Create features
        X, y_class, y_reg = self._create_features(data)

        if len(X) < 20:
            raise ValueError("Not enough data points after feature creation")

        # Train-test split
        split_idx = int(len(X) * (1 - self.test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_class_train, y_class_test = y_class.iloc[:split_idx], y_class.iloc[split_idx:]
        y_reg_train, y_reg_test = y_reg.iloc[:split_idx], y_reg.iloc[split_idx:]

        # Scale features
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index,
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test), columns=X_test.columns, index=X_test.index
        )

        # Train classification model
        self.classifier.fit(X_train_scaled, y_class_train)
        y_class_pred_train = self.classifier.predict(X_train_scaled)
        y_class_pred_test = self.classifier.predict(X_test_scaled)

        classification_accuracy = accuracy_score(y_class_test, y_class_pred_test)

        # Add classification predictions as feature for regression
        X_train_reg = X_train_scaled.copy()
        X_train_reg["direction_pred"] = y_class_pred_train

        X_test_reg = X_test_scaled.copy()
        X_test_reg["direction_pred"] = y_class_pred_test

        # Train regression model
        self.regressor.fit(X_train_reg, y_reg_train)

        # Evaluate regression
        y_reg_pred_test = self.regressor.predict(X_test_reg)
        regression_mae = mean_absolute_error(y_reg_test, y_reg_pred_test)

        # Train ARIMA for comparison (if available)
        arima_mae = None
        if STATSMODELS_AVAILABLE and len(y_reg_train) > 20:
            try:
                arima_mae = self._fit_arima_comparison(y_reg_train, y_reg_test)
            except Exception as e:
                logger.debug(f"ARIMA comparison failed: {e}")

        self.is_fitted = True

        return {
            "classification_accuracy": float(classification_accuracy),
            "regression_mae": float(regression_mae),
            "arima_mae": float(arima_mae) if arima_mae is not None else None,
            "n_train": len(X_train),
            "n_test": len(X_test),
        }

    def _fit_arima_comparison(
        self, train_data: pd.Series, test_data: pd.Series
    ) -> Optional[float]:
        """Fit ARIMA model for comparison."""
        if not STATSMODELS_AVAILABLE:
            return None

        try:
            # Simple ARIMA grid search
            best_aic = np.inf
            best_order = None
            best_model = None

            # Limited grid search for speed
            orders = [(1, 1, 1), (2, 1, 2), (1, 0, 1), (2, 0, 2), (0, 1, 1)]

            for order in orders:
                try:
                    model = ARIMA(train_data, order=order)
                    fitted = model.fit()
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = order
                        best_model = fitted
                except Exception:
                    continue

            if best_model is None:
                return None

            self.arima_model = best_model

            # Forecast
            forecast = best_model.get_forecast(steps=len(test_data))
            forecast_values = forecast.predicted_mean

            mae = mean_absolute_error(test_data, forecast_values)
            return float(mae)

        except Exception as e:
            logger.debug(f"ARIMA fitting failed: {e}")
            return None

    def predict(self, data: Optional[pd.Series] = None, n_steps: int = 1) -> np.ndarray:
        """
        Predict future values.

        Args:
            data: Optional data to use for prediction (uses training data if None)
            n_steps: Number of steps ahead to forecast

        Returns:
            Array of predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        if data is not None:
            # Use provided data for prediction
            X, _, _ = self._create_features(data)
            if len(X) == 0:
                raise ValueError("Not enough data for prediction")

            # Use last row
            X_pred = X.iloc[[-1]].copy()

        else:
            raise ValueError("Must provide data for prediction")

        # Scale features
        X_pred_scaled = pd.DataFrame(
            self.scaler.transform(X_pred), columns=X_pred.columns, index=X_pred.index
        )

        # Predict direction
        direction_pred = self.classifier.predict(X_pred_scaled)

        # Add direction prediction as feature
        X_pred_reg = X_pred_scaled.copy()
        X_pred_reg["direction_pred"] = direction_pred

        # Predict value
        predictions = self.regressor.predict(X_pred_reg)

        # For multi-step ahead, use recursive prediction
        if n_steps > 1:
            full_predictions = []
            current_data = data.copy()

            for step in range(n_steps):
                # Get prediction
                X_step, _, _ = self._create_features(current_data)
                if len(X_step) == 0:
                    break

                X_step_scaled = pd.DataFrame(
                    self.scaler.transform(X_step.iloc[[-1]]),
                    columns=X_step.columns,
                    index=X_step.iloc[[-1]].index,
                )

                dir_pred = self.classifier.predict(X_step_scaled)
                X_step_reg = X_step_scaled.copy()
                X_step_reg["direction_pred"] = dir_pred
                pred_value = self.regressor.predict(X_step_reg)[0]

                full_predictions.append(pred_value)

                # Append prediction to current_data for next iteration
                if hasattr(current_data, "index"):
                    new_index = (
                        current_data.index[-1] + pd.Timedelta(days=1)
                        if pd.api.types.is_datetime64_any_dtype(current_data.index)
                        else len(current_data)
                    )
                else:
                    new_index = len(current_data)

                if isinstance(current_data, pd.Series):
                    current_data = pd.concat(
                        [current_data, pd.Series([pred_value], index=[new_index])]
                    )
                else:
                    current_data = np.append(current_data, pred_value)

            predictions = np.array(full_predictions)

        return predictions

    def fit_predict(
        self,
        data: pd.Series,
        test_size: Optional[float] = None,
        return_metrics: bool = True,
    ) -> ForecastResult:
        """
        Fit model and predict on test set.

        Args:
            data: Full time series data
            test_size: Override test size (uses instance default if None)
            return_metrics: Whether to calculate and return metrics

        Returns:
            ForecastResult with predictions and metrics
        """
        original_test_size = self.test_size
        if test_size is not None:
            self.test_size = test_size

        try:
            # Create features
            X, y_class, y_reg = self._create_features(data)

            split_idx = int(len(X) * (1 - self.test_size))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_class_train, y_class_test = (
                y_class.iloc[:split_idx],
                y_class.iloc[split_idx:],
            )
            y_reg_train, y_reg_test = y_reg.iloc[:split_idx], y_reg.iloc[split_idx:]

            # Fit on training data
            train_metrics = self.fit(data.iloc[: split_idx + self.n_lags + 1])

            # Predict on test set
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index,
            )
            y_class_pred_test = self.classifier.predict(X_test_scaled)
            X_test_reg = X_test_scaled.copy()
            X_test_reg["direction_pred"] = y_class_pred_test
            predictions = self.regressor.predict(X_test_reg)

            # Calculate metrics
            mae = None
            mse = None
            rmse = None
            mape = None
            direction_accuracy = None

            if return_metrics:
                mae = float(mean_absolute_error(y_reg_test, predictions))
                mse = float(mean_squared_error(y_reg_test, predictions))
                rmse = float(np.sqrt(mse))
                mape = float(
                    np.mean(np.abs((y_reg_test - predictions) / (y_reg_test + 1e-10)))
                    * 100
                )
                direction_accuracy = float(
                    accuracy_score(y_class_test, y_class_pred_test)
                )

            # Get timestamps if available
            timestamps = None
            if hasattr(data, "index") and pd.api.types.is_datetime64_any_dtype(
                data.index
            ):
                timestamps = data.index[split_idx + self.n_lags + 1 :]

            return ForecastResult(
                predictions=predictions,
                actual=y_reg_test.values,
                mae=mae,
                mse=mse,
                rmse=rmse,
                mape=mape,
                direction_accuracy=direction_accuracy,
                classification_accuracy=train_metrics.get("classification_accuracy"),
                regression_mae=train_metrics.get("regression_mae"),
                arima_mae=train_metrics.get("arima_mae"),
                timestamps=timestamps,
                n_train=len(X_train),
                n_test=len(X_test),
            )

        finally:
            self.test_size = original_test_size
