"""
Configuration loading module for drift-benchmark - REQ-CFG-XXX

This module defines how BenchmarkConfig is loaded, validated, and processed
using Pydantic v2 validation.
"""

from .config_loader import BenchmarkConfig

__all__ = [
    "BenchmarkConfig",
]
