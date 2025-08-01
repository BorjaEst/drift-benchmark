"""
Results module for drift-benchmark - REQ-RST-XXX

This module provides result management and storage functionality for benchmark results.
"""

from pathlib import Path
from typing import Union

from ..models import BenchmarkResult
from ..settings import get_logger, settings
from .storage import save_benchmark_results

logger = get_logger(__name__)


class Results:
    """
    Results manager class for saving benchmark results.

    This class provides the interface expected by the tests for managing
    benchmark results storage.
    """

    def __init__(self, benchmark_result: BenchmarkResult):
        """Initialize with benchmark result."""
        self.benchmark_result = benchmark_result

    def save(self) -> Path:
        """
        Save the benchmark results to storage.

        Returns:
            Path to the created timestamped directory
        """
        return save_results(self.benchmark_result)


def save_results(benchmark_result: BenchmarkResult) -> Path:
    """
    High-level interface for saving benchmark results.

    This function provides the main entry point for saving benchmark results
    using the configured results directory from settings.

    Args:
        benchmark_result: The benchmark result to save

    Returns:
        Path to the created timestamped directory

    Raises:
        OSError: If directory creation or file writing fails
        ValueError: If benchmark_result is invalid
    """
    try:
        # Use results directory from settings
        results_dir = settings.results_dir

        logger.info(f"Saving benchmark results to: {results_dir}")

        # Use the storage variant
        output_dir = save_benchmark_results(benchmark_result, results_dir)

        return output_dir

    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        raise


# Export both the high-level and low-level interfaces
__all__ = [
    "Results",
    "save_results",
    "save_benchmark_results",
]
