"""
Benchmark module for drift-benchmark - REQ-BEN-XXX

This module contains the benchmark runner to benchmark adapters against each other.
"""

from .benchmark_core import Benchmark
from .benchmark_runner import BenchmarkRunner

__all__ = [
    "Benchmark",
    "BenchmarkRunner",
]
