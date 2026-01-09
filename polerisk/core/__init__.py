"""
Core functionality for soil moisture analysis (part of polerisk).

This module contains the core classes and functions for processing AMSR2 LPRM
soil moisture data and matching it with in-situ measurements. This is used
as foundational data for pole health assessment.
"""

# Import core functions with graceful handling of optional dependencies
try:
    from .lprm_utils import _find_nearest_valid, get_lprm_des, read_lprm_file
    from .matching import _log_processing_summary, match_insitu_with_lprm
except ImportError as e:
    # netCDF4 may not be installed - these functions won't be available
    import warnings
    import sys

    warnings.warn(
        f"Some core functions require netCDF4: {e}. "
        "Install with: pip install netCDF4",
        ImportWarning,
    )

    # Create dummy functions to prevent import errors
    def _find_nearest_valid(*args, **kwargs):
        raise ImportError(
            "netCDF4 is required for this function. Install with: pip install netCDF4"
        )

    def get_lprm_des(*args, **kwargs):
        raise ImportError(
            "netCDF4 is required for this function. Install with: pip install netCDF4"
        )

    def read_lprm_file(*args, **kwargs):
        raise ImportError(
            "netCDF4 is required for this function. Install with: pip install netCDF4"
        )

    def _log_processing_summary(*args, **kwargs):
        raise ImportError(
            "netCDF4 is required for this function. Install with: pip install netCDF4"
        )

    def match_insitu_with_lprm(*args, **kwargs):
        raise ImportError(
            "netCDF4 is required for this function. Install with: pip install netCDF4"
        )


try:
    from ..common.config import ConfigManager

    get_parameters = ConfigManager.get_parameters
except ImportError:

    def get_parameters(*args, **kwargs):
        raise ImportError("ConfigManager not available")


__all__ = [
    "_find_nearest_valid",
    "_log_processing_summary",
    "get_lprm_des",
    "get_parameters",
    "match_insitu_with_lprm",
    "read_lprm_file",
]
