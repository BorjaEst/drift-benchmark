"""
drift-benchmark: A framework for benchmarking drift detection algorithms.

This package provides tools and utilities for benchmarking and comparing
different drift detection algorithms across various datasets and scenarios.
"""

import logging
from typing import Dict, Optional

from drift_benchmark.benchmark import BenchmarkRunner
from drift_benchmark.detectors import (
    BaseDetector,
    discover_and_register_detectors,
    get_detector,
    list_available_detectors,
    register_detector,
)
from drift_benchmark.settings import settings

# Load version from file
try:
    with open("VERSION", "r") as f:
        __version__ = f.read().strip()
except FileNotFoundError:
    __version__ = "0.1.0"  # Default version if VERSION file not found


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def setup(
    components_dir: Optional[str] = None,
    configurations_dir: Optional[str] = None,
    datasets_dir: Optional[str] = None,
    results_dir: Optional[str] = None,
) -> Dict[str, int]:
    """
    Set up the drift-benchmark environment.

    This function configures paths and discovers available detectors.

    Args:
        components_dir: Path to components directory
        configurations_dir: Path to configurations directory
        datasets_dir: Path to datasets directory
        results_dir: Path to results directory

    Returns:
        Dictionary with setup information
    """
    # Update settings if provided
    if any([components_dir, configurations_dir, datasets_dir, results_dir]):
        # Create a dictionary of provided settings
        setting_updates = {}
        if components_dir:
            setting_updates["components_dir"] = components_dir
        if configurations_dir:
            setting_updates["configurations_dir"] = configurations_dir
        if datasets_dir:
            setting_updates["datasets_dir"] = datasets_dir
        if results_dir:
            setting_updates["results_dir"] = results_dir

        # Update settings with provided values
        for key, value in setting_updates.items():
            setattr(settings, key, value)

    # Discover and register detectors
    detector_count = discover_and_register_detectors()

    logger.info(f"drift-benchmark setup complete. Found {detector_count} detectors.")

    return {"detector_count": detector_count}


# Automatically discover detectors when the package is imported
setup()
