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


# Export both the high-level and low-level interfaces
def save_results(benchmark_result: BenchmarkResult, results_dir: Union[str, Path] = "results") -> Path:
    """
    Save benchmark results - alias for save_benchmark_results for backward compatibility.

    Args:
        benchmark_result: The benchmark result to save
        results_dir: Base directory for saving results (default: "results")

    Returns:
        Path to the created timestamped directory
    """
    return save_benchmark_results(benchmark_result, results_dir)


__all__ = ["save_benchmark_results", "save_results"]
