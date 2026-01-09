"""
Enhanced time series visualization utilities with minimalist styling.

Adapted from time_series/WIP visualization utilities, integrated with signalplot
for consistent minimalist aesthetics across polerisk visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from typing import List, Optional, Union
import logging

logger = logging.getLogger(__name__)

# Apply SignalPlot minimalist defaults (complements existing signalplot.apply() calls)
try:
    import signalplot

    signalplot.apply()
except ImportError:
    logger.debug("signalplot not available, using default matplotlib styling")


def set_time_series_style(
    ax, data: pd.DataFrame, time_column: str, value_columns: List[str]
):
    """
    Set minimalist styling for time series plots with dynamic date formatting.

    Parameters:
        ax: matplotlib axes object to style
        data: DataFrame containing time and value columns
        time_column: Name of the time column
        value_columns: List of value column names
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_position(("outward", 5))
    ax.spines["bottom"].set_position(("outward", 5))

    # Ensure datetime format
    if not pd.api.types.is_datetime64_any_dtype(data[time_column]):
        data[time_column] = pd.to_datetime(data[time_column])

    time_range = data[time_column].max() - data[time_column].min()

    # Dynamic X-axis formatting based on time range
    if time_range < pd.Timedelta(days=365):  # Less than 1 year
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_minor_locator(mdates.WeekLocator())
    elif time_range < pd.Timedelta(days=365 * 10):  # Less than 10 years
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_minor_locator(mdates.MonthLocator((1, 7)))  # Jan and Jul
    else:  # 10+ years
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_minor_locator(mdates.YearLocator(1))

    ax.set_xlim(data[time_column].min(), data[time_column].max())

    # Y-axis scaling based on percentiles for better visual distribution
    all_values = np.concatenate([data[col].dropna().values for col in value_columns])
    if len(all_values) > 0:
        y_20, y_mean, y_80 = np.percentile(all_values, [20, 50, 80])
        ax.set_yticks([y_20, y_mean, y_80])
        ax.set_yticklabels([f"{y_20:.2f}", f"{y_mean:.2f}", f"{y_80:.2f}"])

    # Rotate x-axis labels for readability
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")


def plot_time_series_enhanced(
    data: pd.DataFrame,
    time_column: str,
    value_columns: List[str],
    title: str = "Time Series Plot",
    filename: Optional[str] = None,
    figsize: tuple = (10, 5),
    show_labels: bool = True,
) -> plt.Figure:
    """
    Plot time series data with enhanced minimalist styling.

    Parameters:
        data: DataFrame containing time and value data
        time_column: Name of the time column
        value_columns: List of column names to plot
        title: Plot title
        filename: Optional filename to save the plot
        figsize: Figure size tuple
        show_labels: Whether to show labels for each series

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Use grayscale gradient that works well with signalplot
    colors = plt.cm.Greys(np.linspace(0.3, 0.8, len(value_columns)))

    # Ensure datetime format
    data_plot = data.copy()
    if not pd.api.types.is_datetime64_any_dtype(data_plot[time_column]):
        data_plot[time_column] = pd.to_datetime(data_plot[time_column])

    # Plot each series
    for i, col in enumerate(value_columns):
        ax.plot(
            data_plot[time_column],
            data_plot[col],
            linewidth=2,
            color=colors[i],
            label=col,
            alpha=0.8,
        )

        # Add labels at end of series if requested
        if show_labels and len(data_plot) > 0:
            last_x = data_plot[time_column].iloc[-1] + pd.Timedelta(
                days=(
                    time_range.days * 0.02
                    if (
                        time_range := data_plot[time_column].max()
                        - data_plot[time_column].min()
                    )
                    > pd.Timedelta(days=0)
                    else pd.Timedelta(days=10)
                )
            )
            last_y = data_plot[col].iloc[-1]
            ax.text(
                last_x,
                last_y,
                col,
                fontsize=10,
                color=colors[i],
                verticalalignment="center",
                alpha=0.7,
            )

    # Apply styling
    set_time_series_style(ax, data_plot, time_column, value_columns)

    if title:
        ax.set_title(title, fontsize=14, fontweight="normal")
    ax.set_xlabel("Date")
    if len(value_columns) == 1:
        ax.set_ylabel(value_columns[0])
    else:
        ax.legend(loc="best", frameon=False)

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        logger.info(f"Saved plot to {filename}")

    return fig


def plot_decomposition_enhanced(
    data: pd.Series,
    model: str = "additive",
    title: str = "Time Series Decomposition",
    filename: Optional[str] = None,
    period: Optional[int] = None,
    figsize: tuple = (10, 8),
) -> plt.Figure:
    """
    Perform and plot seasonal decomposition with enhanced minimalist styling.

    Parameters:
        data: Time series data to decompose
        model: Type of seasonal component ('additive' or 'multiplicative')
        title: Title for the decomposition plots
        filename: Optional filename to save the plot
        period: Seasonal period (auto-detected if None)
        figsize: Figure size tuple

    Returns:
        matplotlib Figure object
    """
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
    except ImportError:
        logger.warning("statsmodels not available. Using basic decomposition.")
        return _plot_basic_decomposition(data, title, filename, figsize)

    # Auto-detect period if not provided
    if period is None:
        period = max(2, len(data) // 10)
        if len(data) >= 365:
            period = 365  # Annual cycle
        elif len(data) >= 30:
            period = 30  # Monthly cycle
        else:
            period = max(2, len(data) // 4)

    try:
        decomposition = seasonal_decompose(
            data, model=model, period=period, extrapolate_trend="freq"
        )
    except Exception as e:
        logger.warning(
            f"Seasonal decomposition failed: {e}. Using basic decomposition."
        )
        return _plot_basic_decomposition(data, title, filename, figsize)

    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)

    # Original series
    axes[0].plot(
        data.index if hasattr(data, "index") else range(len(data)),
        data.values,
        linewidth=2,
        color="black",
        alpha=0.8,
    )
    axes[0].set_title("Original Series", fontsize=12, fontweight="normal")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[0].set_ylabel("Value")

    # Trend
    trend_clean = decomposition.trend.dropna()
    if len(trend_clean) > 0:
        trend_index = (
            trend_clean.index
            if hasattr(trend_clean, "index")
            else range(len(trend_clean))
        )
        axes[1].plot(
            trend_index, trend_clean.values, linewidth=2, color="black", alpha=0.8
        )
    axes[1].set_title("Trend", fontsize=12, fontweight="normal")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].set_ylabel("Trend")

    # Seasonal
    seasonal_clean = decomposition.seasonal.dropna()
    if len(seasonal_clean) > 0:
        seasonal_index = (
            seasonal_clean.index
            if hasattr(seasonal_clean, "index")
            else range(len(seasonal_clean))
        )
        axes[2].plot(
            seasonal_index,
            seasonal_clean.values,
            linewidth=1.5,
            color="black",
            alpha=0.7,
        )
    axes[2].set_title("Seasonal", fontsize=12, fontweight="normal")
    axes[2].spines["top"].set_visible(False)
    axes[2].spines["right"].set_visible(False)
    axes[2].set_ylabel("Seasonal")

    # Residual
    resid_clean = decomposition.resid.dropna()
    if len(resid_clean) > 0:
        resid_index = (
            resid_clean.index
            if hasattr(resid_clean, "index")
            else range(len(resid_clean))
        )
        axes[3].plot(
            resid_index, resid_clean.values, linewidth=1, color="black", alpha=0.6
        )
        axes[3].axhline(0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    axes[3].set_title("Residual", fontsize=12, fontweight="normal")
    axes[3].spines["top"].set_visible(False)
    axes[3].spines["right"].set_visible(False)
    axes[3].set_ylabel("Residual")
    axes[3].set_xlabel("Time")

    # Format x-axis for datetime if applicable
    if hasattr(data, "index") and pd.api.types.is_datetime64_any_dtype(data.index):
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.suptitle(title, fontsize=14, fontweight="normal", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        logger.info(f"Saved decomposition plot to {filename}")

    return fig


def _plot_basic_decomposition(
    data: pd.Series, title: str, filename: Optional[str], figsize: tuple
) -> plt.Figure:
    """Basic decomposition plot without statsmodels."""
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Original
    axes[0].plot(
        data.index if hasattr(data, "index") else range(len(data)),
        data.values,
        linewidth=2,
        color="black",
        alpha=0.8,
    )
    axes[0].set_title("Original Series", fontsize=12, fontweight="normal")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # Simple moving average as trend approximation
    window = min(30, len(data) // 4)
    if window > 1:
        trend = data.rolling(window=window, center=True).mean()
        residual = data - trend
        axes[1].plot(
            trend.index if hasattr(trend, "index") else range(len(trend)),
            trend.values,
            linewidth=2,
            color="black",
            alpha=0.8,
            label="Trend (MA)",
        )
        axes[1].fill_between(
            residual.index if hasattr(residual, "index") else range(len(residual)),
            residual.values,
            alpha=0.3,
            label="Residual",
        )
    else:
        axes[1].plot(
            data.index if hasattr(data, "index") else range(len(data)),
            data.values,
            linewidth=2,
            color="black",
            alpha=0.8,
        )

    axes[1].set_title("Trend & Residual", fontsize=12, fontweight="normal")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].legend(frameon=False)

    plt.suptitle(title, fontsize=14, fontweight="normal", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")

    return fig
