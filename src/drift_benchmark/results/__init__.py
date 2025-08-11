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
__all__ = ["save_benchmark_results"]
