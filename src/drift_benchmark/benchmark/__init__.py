"""
Benchmark module for drift-benchmark - REQ-BEN-XXX

This module contains the benchmark runner to benchmark adapters against each other.
"""

# Export logger for test mocking
from ..settings import get_logger
from .core import Benchmark
from .runner import BenchmarkRunner

logger = get_logger(__name__)

__all__ = [
    "Benchmark",
    "BenchmarkRunner",
]
