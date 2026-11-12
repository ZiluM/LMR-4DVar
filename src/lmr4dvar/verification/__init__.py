"""Time-series and spatial-pattern verification."""

from .metrics import (
    SeriesMetrics,
    anomaly,
    centered_running_mean,
    gridpoint_temporal_correlation,
    seasonal_anomaly,
    series_metrics,
    spatial_pattern_metrics,
)
from .plotting import plot_correlation_maps, plot_index_comparison

__all__ = [
    "SeriesMetrics",
    "anomaly",
    "centered_running_mean",
    "gridpoint_temporal_correlation",
    "seasonal_anomaly",
    "series_metrics",
    "spatial_pattern_metrics",
    "plot_correlation_maps",
    "plot_index_comparison",
]
