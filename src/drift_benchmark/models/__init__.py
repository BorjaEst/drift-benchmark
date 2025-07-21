"""
Data models for drift-benchmark using Pydantic v2 - REQ-MOD-XXX

This module contains data models used throughout the drift-benchmark library
for type safety and validation.
"""

from .configurations import BenchmarkConfig, DatasetConfig, DetectorConfig
from .metadata import BenchmarkSummary, DatasetMetadata, DetectorMetadata
from .results import BenchmarkResult, DatasetResult, DetectorResult

__all__ = [
    # Configuration models
    "BenchmarkConfig",
    "DatasetConfig",
    "DetectorConfig",
    # Metadata models
    "DatasetMetadata",
    "DetectorMetadata",
    "BenchmarkSummary",
    # Result models
    "DatasetResult",
    "DetectorResult",
    "BenchmarkResult",
]
